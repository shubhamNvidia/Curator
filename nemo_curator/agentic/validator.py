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
"""Layer 2 — deterministic static IR validator.

Eight checks, none of which use the LLM:

1. **Stage resolution** — every ``StageRef.stage`` resolves to a registry entry.
2. **Param surface** — keys in ``StageRef.params`` are recognized; numeric
   bounds (min/max) and enum choices are honored.
3. **Key flow** — produced keys flow to consumers; producers exist for every
   declared input.
4. **Preconditions with auto-insert** — when a stage declares
   ``requires_sample_rate`` and the upstream does not produce that SR, the
   validator inserts ``MonoConversionStage`` (48 kHz) or ``ResampleAudioStage``
   (16 kHz) ahead of it.
5. **License gate** — if ``IntentCategories.policy.commercial_only`` is true,
   every stage / model in the IR must be ``commercial_safe``.
6. **Resource sanity** — ``gpus`` and ``gpu_memory_gb`` cannot both be > 0.
7. **Read/write boundary** — the first stage must be a source (``READ_MANIFEST``
   or ``DATASET_CREATE``) and the last must be a sink (``WRITE_MANIFEST``).
8. **Cardinality** — :class:`MANY_TO_ONE` stages are only allowed adjacent to
   the sink; :class:`ONE_TO_MANY` stages must come *after* any source.

If any blocking finding is emitted the validator refuses to compile; if only
:class:`Severity.WARNING` rows are emitted the IR is still compileable. The
``mutate=True`` mode returns the auto-fixed IR; ``mutate=False`` returns the
original IR plus a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nemo_curator.agentic.cards import (
    COMMERCIAL_OK,
    PHASE_ORDER,
    CapabilityTag,
    Cardinality,
    InputShape,
    LicenseKind,
    OutputShape,
    Phase,
    ResourceSpec,
    StageCard,
)
from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.ir import PipelineIR, StageRef
from nemo_curator.agentic.registry import CapabilityRegistry


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


class Severity(str, Enum):
    """How blocking a finding is."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """One row of the validator's report."""

    severity: Severity
    code: str
    detail: str
    stage_index: int | None = None
    stage_name: str | None = None


@dataclass
class ValidationReport:
    """All findings + the (possibly auto-fixed) IR."""

    ir: PipelineIR
    findings: list[Finding] = field(default_factory=list)
    auto_inserted: list[StageRef] = field(default_factory=list)
    fingerprint: str = ""

    def is_ok(self) -> bool:
        return not any(f.severity == Severity.ERROR for f in self.findings)

    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]


# ----------------------------------------------------------------------------
# Auto-insert decision
# ----------------------------------------------------------------------------


MONO_STAGE = "MonoConversionStage"
RESAMPLE_STAGE = "ResampleAudioStage"


def _auto_insert_for_sample_rate(target_sr: int, sink_target_dir: str | None) -> list[StageRef]:
    """Return the stage(s) needed to deliver target_sr mono in-memory waveform.

    Semantics summary:

    - ``ResampleAudioStage`` is the only stage that ACTUALLY resamples; it
      runs ffmpeg on-disk and writes a new file. We point its output dir at
      ``<sink_target_dir>/_resampled`` so artifacts stay co-located with the
      run, and overwrite ``audio_filepath`` so downstream stages remain
      transparent.
    - ``MonoConversionStage`` does channel-collapse + SR *verification*;
      with ``strict_sample_rate=true`` it DROPS files whose SR doesn't
      match. Because Resample runs first, every survivor matches.

    We always emit both stages so the planner doesn't have to reason about
    whether the source corpus is heterogeneous — a uniform corpus pays a
    tiny ffmpeg re-encode cost but the contract becomes deterministic.
    """

    resampled_dir = (
        f"{sink_target_dir.rstrip('/')}/_resampled"
        if sink_target_dir
        else "/tmp/adv_resampled"
    )
    resample = StageRef(
        stage=RESAMPLE_STAGE,
        params={
            "resampled_audio_dir": resampled_dir,
            "target_sample_rate": target_sr,
            "target_nchannels": 1,
            "target_format": "wav",
            # Overwrite the standard key so downstream finds the resampled file.
            "resampled_audio_filepath_key": "audio_filepath",
        },
        auto_inserted=True,
        insert_reason=f"normalize SR + channels to {target_sr} mono",
    )
    mono = StageRef(
        stage=MONO_STAGE,
        params={"output_sample_rate": target_sr, "strict_sample_rate": True},
        auto_inserted=True,
        insert_reason=f"load mono in-memory waveform at {target_sr}",
    )
    return [resample, mono]


# ----------------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------------


def validate(
    ir: PipelineIR,
    registry: CapabilityRegistry,
    *,
    intent: IntentCategories | None = None,
    mutate: bool = True,
) -> ValidationReport:
    """Run the eight checks. Returns a :class:`ValidationReport`."""

    intent = intent or ir.intent or IntentCategories()
    findings: list[Finding] = []
    inserted: list[StageRef] = []

    # Work on a copy so the original IR object is not silently mutated.
    working_ir = ir.model_copy(deep=True)

    # --- Check 1: stage resolution -----------------------------------------
    _check_stage_resolution(working_ir, registry, findings)
    if any(f.severity == Severity.ERROR for f in findings):
        return ValidationReport(ir=working_ir, findings=findings, auto_inserted=inserted)

    # --- Check 6: resource sanity (cheap; run early) -----------------------
    _check_resource_sanity(working_ir, findings)

    # --- Check 2: params --------------------------------------------------
    _check_params(working_ir, registry, findings)

    # --- Check 5: license gate --------------------------------------------
    _check_license_gate(working_ir, registry, intent, findings)

    # --- Check 4: preconditions / auto-insert ------------------------------
    if mutate:
        _apply_autoinsert(working_ir, registry, findings, inserted)
    else:
        _check_preconditions_only(working_ir, registry, findings)

    # --- Check 4a: phase-aware coarse reorder ----------------------------
    # Hard pipeline-phase ordering ("source < preprocess < load < segment
    # < analyze < concat < package < polish < materialize < sink").
    # Stable rank-sort runs BEFORE the data-key topo step so producer/
    # consumer logic operates on an already phase-correct sequence.
    _phase_reorder(working_ir, registry, findings, mutate=mutate)

    # --- Check 4b: topological reorder by data-key dependencies -----------
    # Producer-before-consumer is a HARD requirement; when ``mutate`` is on
    # we silently fix any stage whose required key is only produced later.
    _topo_sort_stages(working_ir, registry, findings, mutate=mutate)

    # --- Check 4c: shape compatibility / segment-concat auto-insert ------
    # If an upstream stage emits FAN_OUT_SEGMENTS or FAN_OUT_SPEAKERS and a
    # downstream stage requires WHOLE_FILE shape, we insert
    # SegmentConcatenationStage in-between so each source-file is
    # re-stitched before the consumer sees it.
    if mutate:
        _apply_shape_autoinsert(working_ir, registry, findings, inserted)

    # --- Check 3: key flow -------------------------------------------------
    _check_key_flow(working_ir, registry, findings)

    # --- Check 3b: soft ordering preferences ------------------------------
    _check_ordering_hints(working_ir, registry, findings)

    # --- Check 7: read/write boundary --------------------------------------
    _check_boundary(working_ir, registry, findings)

    # --- Check 8: cardinality ----------------------------------------------
    _check_cardinality(working_ir, registry, findings)

    # --- Check 9: terminal stage placement ---------------------------------
    # A stage marked ``terminal: true`` clears task.data or finalizes the
    # row schema; nothing audio-analytical may run after it.
    _check_terminal_placement(working_ir, registry, findings)

    return ValidationReport(
        ir=working_ir,
        findings=findings,
        auto_inserted=inserted,
        fingerprint=_fingerprint(working_ir),
    )


# ----------------------------------------------------------------------------
# Check implementations
# ----------------------------------------------------------------------------


def _check_stage_resolution(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    for i, s in enumerate(ir.stages):
        if s.stage not in reg:
            findings.append(Finding(
                severity=Severity.ERROR,
                code="stage_not_registered",
                detail=f"Stage {s.stage!r} not found in the capability registry.",
                stage_index=i,
                stage_name=s.stage,
            ))


def _check_resource_sanity(ir: PipelineIR, findings: list[Finding]) -> None:
    for i, s in enumerate(ir.stages):
        if s.resources and s.resources.gpus > 0 and s.resources.gpu_memory_gb > 0:
            findings.append(Finding(
                severity=Severity.ERROR,
                code="gpu_xor_violated",
                detail=(
                    f"{s.stage}: gpus={s.resources.gpus} and gpu_memory_gb={s.resources.gpu_memory_gb} "
                    "are mutually exclusive."
                ),
                stage_index=i,
                stage_name=s.stage,
            ))


def _check_params(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        if entry is None:
            continue
        known = {p.name: p for p in entry.card.params}
        for k, v in s.params.items():
            spec = known.get(k)
            if spec is None:
                findings.append(Finding(
                    severity=Severity.WARNING,
                    code="unknown_param",
                    detail=f"{s.stage}.params[{k!r}] is not in the card; it will be passed through as a kwarg.",
                    stage_index=i,
                    stage_name=s.stage,
                ))
                continue
            if spec.choices is not None and v not in spec.choices and v is not None:
                findings.append(Finding(
                    severity=Severity.ERROR,
                    code="param_choice_violated",
                    detail=f"{s.stage}.{k}={v!r} is not in choices={spec.choices}",
                    stage_index=i,
                    stage_name=s.stage,
                ))
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if spec.min is not None and v < spec.min:
                    findings.append(Finding(
                        severity=Severity.ERROR,
                        code="param_under_min",
                        detail=f"{s.stage}.{k}={v} < min={spec.min}",
                        stage_index=i,
                        stage_name=s.stage,
                    ))
                if spec.max is not None and v > spec.max:
                    findings.append(Finding(
                        severity=Severity.ERROR,
                        code="param_over_max",
                        detail=f"{s.stage}.{k}={v} > max={spec.max}",
                        stage_index=i,
                        stage_name=s.stage,
                    ))


def _check_license_gate(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    intent: IntentCategories,
    findings: list[Finding],
) -> None:
    if not intent.policy.commercial_only:
        return
    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        if entry is None:
            continue
        card = entry.card
        if not card.commercial_safe:
            findings.append(Finding(
                severity=Severity.ERROR,
                code="commercial_only_violation",
                detail=f"{s.stage}: card.commercial_safe=false but intent.policy.commercial_only=true.",
                stage_index=i,
                stage_name=s.stage,
            ))
            continue
        for m in card.models:
            if m.license not in COMMERCIAL_OK:
                findings.append(Finding(
                    severity=Severity.ERROR,
                    code="commercial_only_model_block",
                    detail=(
                        f"{s.stage} model {m.name!r} has license {m.license.value}, "
                        "which is not in the commercial-OK allowlist."
                    ),
                    stage_index=i,
                    stage_name=s.stage,
                ))


def _apply_autoinsert(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
    inserted_out: list[StageRef],
) -> None:
    """Auto-insert Mono / Resample / source-reader stages with unmet preconditions.

    Three precondition shapes are honored:

    1. **Source reader**: if the first stage does not declare
       :class:`CapabilityTag.READ_MANIFEST` or
       :class:`CapabilityTag.DATASET_CREATE`, a reader is prepended via
       :func:`nemo_curator.agentic.adapters.reader_stage`. This is the
       counterpart to the boundary check downstream; without it the agent
       has to remember to add ``ManifestReader`` explicitly.
    2. ``requires_sample_rate: <int>`` — prepend the matching normalizer
       (``MonoConversionStage`` for 48 kHz, ``ResampleAudioStage`` otherwise).
    3. ``requires_in_memory_waveform: true`` with no upstream
       ``waveform`` / ``sample_rate`` producer — prepend
       ``MonoConversionStage`` at 48 kHz so the in-memory tensor exists.
    """

    # --- (1) Source reader prepend ----------------------------------------
    if ir.stages:
        head = reg.get(ir.stages[0].stage)
        head_caps: set[CapabilityTag] = set()
        if head is not None:
            head_caps = set(head.card.capabilities) | set(head.card.also_handles)
        if not head_caps & {CapabilityTag.READ_MANIFEST, CapabilityTag.DATASET_CREATE}:
            try:
                from nemo_curator.agentic.adapters import reader_stage  # noqa: PLC0415

                reader = reader_stage(ir.source)
            except Exception as exc:  # noqa: BLE001
                findings.append(Finding(
                    severity=Severity.ERROR,
                    code="missing_source",
                    detail=(
                        f"First stage {ir.stages[0].stage!r} is not a source and a "
                        f"reader could not be auto-inserted from source spec: {exc}"
                    ),
                    stage_index=0,
                    stage_name=ir.stages[0].stage,
                ))
            else:
                ir.stages.insert(0, reader)
                inserted_out.append(reader)
                findings.append(Finding(
                    severity=Severity.INFO,
                    code="auto_inserted",
                    detail=(
                        f"Auto-inserted {reader.stage} as the source-reader "
                        f"(derived from source.kind={ir.source.kind!r})."
                    ),
                    stage_index=0,
                    stage_name=reader.stage,
                ))

    # --- (1b) Sink writer append -----------------------------------------
    if ir.stages:
        tail = reg.get(ir.stages[-1].stage)
        tail_caps: set[CapabilityTag] = set()
        if tail is not None:
            tail_caps = set(tail.card.capabilities) | set(tail.card.also_handles)
        if not tail_caps & {CapabilityTag.WRITE_MANIFEST}:
            sink_path = (
                f"{ir.sink.target_dir.rstrip('/')}/{ir.sink.manifest_filename}"
                if ir.sink.target_dir else "manifest.jsonl"
            )
            writer = StageRef(
                stage="ManifestWriterStage",
                params={"output_path": sink_path},
                auto_inserted=True,
                insert_reason="sink=manifest_writer",
            )
            ir.stages.append(writer)
            inserted_out.append(writer)
            findings.append(Finding(
                severity=Severity.INFO,
                code="auto_inserted",
                detail=(
                    f"Auto-inserted {writer.stage} as the sink-writer "
                    f"(target={sink_path!r})."
                ),
                stage_index=len(ir.stages) - 1,
                stage_name=writer.stage,
            ))

    # --- (1c) TimestampMapper before SegmentExtraction --------------------
    # SegmentExtractionStage requires original_start_ms / original_end_ms,
    # which TimestampMapperStage is the canonical producer of. If the agent
    # forgot to add it, we insert it directly upstream of the first
    # SegmentExtraction so the extraction has timestamps to work from.
    _TIMESTAMP_KEYS = {"original_start_ms", "original_end_ms"}
    inserted_indices: set[int] = set()
    i = 0
    while i < len(ir.stages):
        s = ir.stages[i]
        if s.stage == "SegmentExtractionStage":
            # Are the timestamp keys already produced by some stage at index < i?
            available: set[str] = set(_INITIAL_KEYS)
            for upstream in ir.stages[:i]:
                up_entry = reg.get(upstream.stage)
                if up_entry is not None:
                    available |= set(up_entry.card.outputs.data)
                    available |= set(up_entry.card.produces_keys_after_run)
            if not _TIMESTAMP_KEYS.issubset(available):
                tm = StageRef(
                    stage="TimestampMapperStage",
                    params={},
                    auto_inserted=True,
                    insert_reason="produce original_*_ms for SegmentExtractionStage",
                )
                ir.stages.insert(i, tm)
                inserted_out.append(tm)
                findings.append(Finding(
                    severity=Severity.INFO,
                    code="auto_inserted",
                    detail=(
                        "Auto-inserted TimestampMapperStage before "
                        "SegmentExtractionStage to satisfy "
                        "original_start_ms / original_end_ms inputs."
                    ),
                    stage_index=i,
                    stage_name="TimestampMapperStage",
                ))
                inserted_indices.add(i)
                i += 2
                continue
        i += 1

    # --- (2)/(3) Mono / Resample preconditions ----------------------------
    new_stages: list[StageRef] = []
    upstream_sr: int | None = None
    has_in_memory_waveform = False

    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        card: StageCard | None = entry.card if entry else None

        needs_sr = card.requires_sample_rate if card else None
        needs_waveform = bool(card and card.requires_in_memory_waveform)

        if card is not None and needs_sr and upstream_sr != needs_sr:
            mutated = _retune_trailing_normalizer_chain(new_stages, needs_sr)
            if mutated:
                findings.append(Finding(
                    severity=Severity.INFO,
                    code="auto_inserted",
                    detail=(
                        f"Retuned existing trailing {RESAMPLE_STAGE}/{MONO_STAGE} "
                        f"chain to target_sr={needs_sr} for {s.stage} instead of "
                        "inserting a duplicate pair."
                    ),
                    stage_index=i,
                    stage_name=s.stage,
                ))
                upstream_sr = needs_sr
                has_in_memory_waveform = any(
                    ref.stage == MONO_STAGE for ref in _trailing_normalizers(new_stages)
                ) or has_in_memory_waveform
            else:
                chain = _auto_insert_for_sample_rate(needs_sr, ir.sink.target_dir)
                for ins in chain:
                    new_stages.append(ins)
                    inserted_out.append(ins)
                findings.append(Finding(
                    severity=Severity.INFO,
                    code="auto_inserted",
                    detail=(
                        f"Auto-inserted [{' -> '.join(c.stage for c in chain)}] "
                        f"before {s.stage} (target_sr={needs_sr})."
                    ),
                    stage_index=i,
                    stage_name=s.stage,
                ))
                upstream_sr = needs_sr
                has_in_memory_waveform = any(c.stage == MONO_STAGE for c in chain)
        elif needs_waveform and not has_in_memory_waveform:
            # Only insert a fresh chain if the tail isn't already a normalizer
            # we can promote (e.g. the planner-emitted Resample without a
            # following Mono). Promoting means appending a Mono so the
            # in-memory waveform contract is satisfied without doubling up.
            promoted = _promote_trailing_resample_to_full_chain(
                new_stages, target_sr=upstream_sr or 48000
            )
            if promoted is not None:
                new_stages.append(promoted)
                inserted_out.append(promoted)
                findings.append(Finding(
                    severity=Severity.INFO,
                    code="auto_inserted",
                    detail=(
                        f"Auto-inserted {promoted.stage} after the existing "
                        f"{RESAMPLE_STAGE} so {s.stage} has an in-memory "
                        "waveform without duplicating resampling."
                    ),
                    stage_index=i,
                    stage_name=s.stage,
                ))
                upstream_sr = int(promoted.params.get("output_sample_rate", 48000))
                has_in_memory_waveform = True
            else:
                chain = _auto_insert_for_sample_rate(48000, ir.sink.target_dir)
                for ins in chain:
                    new_stages.append(ins)
                    inserted_out.append(ins)
                findings.append(Finding(
                    severity=Severity.INFO,
                    code="auto_inserted",
                    detail=(
                        f"Auto-inserted [{' -> '.join(c.stage for c in chain)}] "
                        f"before {s.stage} (in-memory waveform required)."
                    ),
                    stage_index=i,
                    stage_name=s.stage,
                ))
                upstream_sr = 48000
                has_in_memory_waveform = True

        new_stages.append(s)
        # Update upstream knowledge based on this stage's behavior.
        if s.stage == MONO_STAGE:
            upstream_sr = int(s.params.get("output_sample_rate", 48000))
            has_in_memory_waveform = True
        elif s.stage == RESAMPLE_STAGE:
            upstream_sr = int(s.params.get("target_sample_rate", 16000))
        elif card and card.produces_in_memory_waveform:
            has_in_memory_waveform = True

    ir.stages = new_stages


def _trailing_normalizers(stages: list[StageRef]) -> list[StageRef]:
    """Return the contiguous run of Resample/Mono stages at the tail of ``stages``.

    The run is returned in pipeline order. An empty list means the tail
    is not a normalizer block, so it's not safe to mutate.
    """

    tail: list[StageRef] = []
    for ref in reversed(stages):
        if ref.stage in (RESAMPLE_STAGE, MONO_STAGE):
            tail.append(ref)
        else:
            break
    return list(reversed(tail))


def _retune_trailing_normalizer_chain(
    stages: list[StageRef],
    target_sr: int,
) -> bool:
    """Mutate the trailing Resample/Mono block to target ``target_sr``.

    Returns ``True`` when at least one normalizer's target SR was updated.
    """

    tail = _trailing_normalizers(stages)
    if not tail:
        return False
    mutated = False
    for ref in tail:
        if ref.stage == RESAMPLE_STAGE:
            current = ref.params.get("target_sample_rate")
            if current != target_sr:
                ref.params = {**ref.params, "target_sample_rate": target_sr}
                mutated = True
        elif ref.stage == MONO_STAGE:
            current = ref.params.get("output_sample_rate")
            if current != target_sr:
                ref.params = {**ref.params, "output_sample_rate": target_sr}
                mutated = True
    return mutated


def _promote_trailing_resample_to_full_chain(
    stages: list[StageRef],
    *,
    target_sr: int,
) -> StageRef | None:
    """If the tail is a lone Resample (no Mono after), return the Mono to append.

    Returns ``None`` when promotion is not applicable (no trailing Resample,
    or a Mono is already there).
    """

    tail = _trailing_normalizers(stages)
    if not tail:
        return None
    if any(ref.stage == MONO_STAGE for ref in tail):
        return None
    last_resample = next(
        (ref for ref in reversed(tail) if ref.stage == RESAMPLE_STAGE), None
    )
    if last_resample is None:
        return None
    sr = int(last_resample.params.get("target_sample_rate", target_sr))
    return StageRef(
        stage=MONO_STAGE,
        params={"output_sample_rate": sr, "strict_sample_rate": True},
        auto_inserted=True,
        insert_reason=(
            f"load mono in-memory waveform at {sr} (promoted alongside existing "
            f"{RESAMPLE_STAGE})"
        ),
    )


def _check_preconditions_only(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    """When mutate=False, raise findings for unsatisfied requires_sample_rate."""

    upstream_sr: int | None = None
    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        card = entry.card if entry else None
        if card is not None and card.requires_sample_rate and upstream_sr != card.requires_sample_rate:
            findings.append(Finding(
                severity=Severity.ERROR,
                code="precondition_unsatisfied",
                detail=(
                    f"{s.stage} requires sample_rate={card.requires_sample_rate} but upstream "
                    f"provides {upstream_sr!r}. Re-run with mutate=True to auto-insert."
                ),
                stage_index=i,
                stage_name=s.stage,
            ))
        if s.stage == MONO_STAGE:
            upstream_sr = int(s.params.get("output_sample_rate", 48000))
        elif s.stage == RESAMPLE_STAGE:
            upstream_sr = int(s.params.get("target_sample_rate", 16000))


def _check_key_flow(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    """Track produced data keys forward and complain if a consumer reads one that was never written."""

    available: set[str] = set()
    # Common keys that always exist in fresh AudioTasks
    available.update({"audio_filepath", "task_id"})

    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        if entry is None:
            continue
        card = entry.card
        for needed in card.inputs.data:
            if needed not in available:
                findings.append(Finding(
                    severity=Severity.WARNING,
                    code="missing_data_key",
                    detail=(
                        f"{s.stage} expects task.data[{needed!r}] but no upstream stage declares it. "
                        "If this key comes from the manifest itself, the warning is benign."
                    ),
                    stage_index=i,
                    stage_name=s.stage,
                ))
        for produced in _produced_data_keys(card, s):
            available.add(produced)
        for produced in card.produces_keys_after_run:
            available.add(produced)
        for dropped in card.drops_keys_after_run:
            available.discard(dropped)


_INITIAL_KEYS: frozenset[str] = frozenset({"audio_filepath", "task_id"})


def _produced_data_keys(card: StageCard, ref: StageRef) -> set[str]:
    """Data keys produced by a stage, including param-dependent outputs."""

    produced = set(card.outputs.data)
    if (
        card.name == "VADSegmentationStage"
        and bool(ref.params.get("nested"))
        and card.nested_segment_key
    ):
        produced.add(card.nested_segment_key)
    return produced


def _topo_sort_stages(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
    *,
    mutate: bool,
) -> None:
    """Reorder stages only when the user's order has a real producer-after-consumer
    violation. We deliberately *preserve* the user's intended order whenever it
    is data-flow-valid: when both stage A and stage B can produce key K, we do
    not add a spurious dependency just because of that overlap.

    Algorithm:

    1. Walk the user's order, tracking ``available`` keys (seeded with
       :data:`_INITIAL_KEYS`).
    2. For each consumer C at position i, find any required key K that is NOT
       yet ``available`` AND that some stage at position j > i produces.
       Only then add an edge j → i.
    3. Run Kahn's with stable tie-break on user's index, so an already-valid
       order is left alone.

    Failures:

    - Cycle: emit ``Severity.ERROR`` and leave the IR alone.
    - Truly unsatisfiable key (no producer anywhere): leave to
      ``_check_key_flow`` to warn; we do nothing.
    """

    n = len(ir.stages)
    if n < 2:
        return

    cards: list[StageCard | None] = []
    produces_sets: list[set[str]] = []
    requires_sets: list[set[str]] = []
    for s in ir.stages:
        entry = reg.get(s.stage)
        card = entry.card if entry else None
        cards.append(card)
        if card is None:
            produces_sets.append(set())
            requires_sets.append(set())
            continue
        produces_sets.append(
            _produced_data_keys(card, s) | set(card.produces_keys_after_run)
        )
        requires_sets.append(set(card.inputs.data))

    # Only add edges for keys NOT already satisfied by the user's order.
    edges: list[set[int]] = [set() for _ in range(n)]
    in_degree = [0] * n
    available: set[str] = set(_INITIAL_KEYS)
    for i in range(n):
        for needed in requires_sets[i]:
            if needed in available:
                continue
            # Find any later stage in user's order that produces this key.
            for j in range(i + 1, n):
                if needed in produces_sets[j]:
                    if i not in edges[j]:
                        edges[j].add(i)
                        in_degree[i] += 1
                    break
        # Update availability for subsequent stages (per user's intended order).
        available |= produces_sets[i]

    if not any(edges):
        return  # user's order is already producer-before-consumer.

    import heapq

    heap: list[int] = [i for i in range(n) if in_degree[i] == 0]
    heapq.heapify(heap)
    new_order: list[int] = []
    while heap:
        i = heapq.heappop(heap)
        new_order.append(i)
        for nb in edges[i]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                heapq.heappush(heap, nb)

    if len(new_order) < n:
        findings.append(Finding(
            severity=Severity.ERROR,
            code="ordering_cycle",
            detail=(
                "Cannot topologically order the pipeline — a stage's required "
                "data key is also produced downstream of it. Inspect "
                "inputs/outputs on the cards involved."
            ),
        ))
        return

    if new_order == list(range(n)):
        return

    if not mutate:
        findings.append(Finding(
            severity=Severity.ERROR,
            code="stage_order_violation",
            detail=(
                "Stages are not in producer-before-consumer order. Re-run with "
                "mutate=True to let the validator reorder, or fix the IR."
            ),
        ))
        return

    new_stages = [ir.stages[i] for i in new_order]
    swaps = [(i, new_order[i]) for i in range(n) if new_order[i] != i]
    ir.stages = new_stages
    findings.append(Finding(
        severity=Severity.INFO,
        code="stage_reordered",
        detail=(
            f"Reordered {len(swaps)} stage(s) to honor data-flow dependencies. "
            f"New order: [{', '.join(s.stage for s in ir.stages)}]."
        ),
    ))


def _check_ordering_hints(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    """Emit INFO findings when a stage's soft ordering preferences are violated.

    A reference in ``prefer_after`` / ``prefer_before`` can be either a stage
    class name (e.g. ``"VADSegmentationStage"``) or a capability tag string
    (e.g. ``"vad"``). Capability references match any registered stage that
    declares the tag in its ``capabilities`` or ``also_handles`` list.
    """

    # Resolve each stage's position and the capability set of every stage in
    # the IR so we can match preferences by either name or tag.
    positions: dict[str, int] = {s.stage: i for i, s in enumerate(ir.stages)}
    caps_by_index: list[set[str]] = []
    for s in ir.stages:
        entry = reg.get(s.stage)
        if entry is None:
            caps_by_index.append(set())
            continue
        caps = (
            {t.value for t in entry.card.capabilities}
            | {t.value for t in entry.card.also_handles}
        )
        caps_by_index.append(caps)

    def _matches(ref: str) -> list[int]:
        # Stage-name match takes precedence; otherwise capability tag.
        if ref in positions:
            return [positions[ref]]
        return [i for i, caps in enumerate(caps_by_index) if ref in caps]

    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        if entry is None:
            continue
        oh = entry.card.selection_hints.ordering_hints
        for ref in oh.prefer_after:
            for j in _matches(ref):
                if j > i:
                    findings.append(Finding(
                        severity=Severity.INFO,
                        code="ordering_preference",
                        detail=(
                            f"{s.stage} prefers to run AFTER {ref!r} "
                            f"but {ir.stages[j].stage} (index {j}) is currently "
                            f"downstream of it (index {i})."
                            + (f" Rationale: {oh.rationale}" if oh.rationale else "")
                        ),
                        stage_index=i,
                        stage_name=s.stage,
                    ))
        for ref in oh.prefer_before:
            for j in _matches(ref):
                if j < i:
                    findings.append(Finding(
                        severity=Severity.INFO,
                        code="ordering_preference",
                        detail=(
                            f"{s.stage} prefers to run BEFORE {ref!r} "
                            f"but {ir.stages[j].stage} (index {j}) is currently "
                            f"upstream of it (index {i})."
                            + (f" Rationale: {oh.rationale}" if oh.rationale else "")
                        ),
                        stage_index=i,
                        stage_name=s.stage,
                    ))


def _check_boundary(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    if not ir.stages:
        findings.append(Finding(
            severity=Severity.ERROR,
            code="empty_pipeline",
            detail="The IR has no stages.",
        ))
        return

    head = reg.get(ir.stages[0].stage)
    tail = reg.get(ir.stages[-1].stage)

    if head is not None:
        head_caps = set(head.card.capabilities) | set(head.card.also_handles)
        if not head_caps & {CapabilityTag.READ_MANIFEST, CapabilityTag.DATASET_CREATE}:
            findings.append(Finding(
                severity=Severity.WARNING,
                code="missing_source",
                detail=f"First stage {ir.stages[0].stage!r} does not declare READ_MANIFEST or DATASET_CREATE.",
                stage_index=0,
                stage_name=ir.stages[0].stage,
            ))

    if tail is not None:
        tail_caps = set(tail.card.capabilities) | set(tail.card.also_handles)
        if CapabilityTag.WRITE_MANIFEST not in tail_caps and CapabilityTag.AUDIO_TO_DOCUMENT not in tail_caps:
            findings.append(Finding(
                severity=Severity.WARNING,
                code="missing_sink",
                detail=f"Last stage {ir.stages[-1].stage!r} does not declare WRITE_MANIFEST.",
                stage_index=len(ir.stages) - 1,
                stage_name=ir.stages[-1].stage,
            ))


def _check_cardinality(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        if entry is None:
            continue
        card = entry.card
        if card.produces_cardinality == Cardinality.MANY_TO_ONE and i != len(ir.stages) - 1:
            findings.append(Finding(
                severity=Severity.ERROR,
                code="fan_in_not_at_end",
                detail=f"{s.stage} is N:1 (fan-in) but is not the last stage; it must be adjacent to the sink.",
                stage_index=i,
                stage_name=s.stage,
            ))


# ----------------------------------------------------------------------------
# Phase-aware reorder
# ----------------------------------------------------------------------------


def _phase_reorder(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
    *,
    mutate: bool,
) -> None:
    """Stable-sort stages by their card's :class:`Phase`.

    Phases form a *hard* ordering constraint:

        source < preprocess < load < segment < analyze < concat <
        package < polish < materialize < sink

    A stage with a smaller phase index MUST run before any stage with a
    larger phase index. Within the same phase the user's relative order is
    preserved (the sort is stable), and downstream data-key topo logic
    handles fine-grained sequencing.

    Stages whose cards omit ``phase`` (default :class:`Phase.ANALYZE`)
    still get a rank, so legacy cards do not break the sort.
    """

    n = len(ir.stages)
    if n < 2:
        return

    def _rank(stage_name: str) -> int:
        entry = reg.get(stage_name)
        if entry is None:
            return PHASE_ORDER[Phase.ANALYZE]
        return PHASE_ORDER.get(entry.card.phase, PHASE_ORDER[Phase.ANALYZE])

    indexed = list(enumerate(ir.stages))
    indexed.sort(key=lambda pair: (_rank(pair[1].stage), pair[0]))
    new_order = [orig_i for orig_i, _ in indexed]

    if new_order == list(range(n)):
        return

    if not mutate:
        findings.append(Finding(
            severity=Severity.ERROR,
            code="phase_order_violation",
            detail=(
                "Stages are not in phase order "
                "(source < preprocess < load < segment < analyze < concat "
                "< package < polish < materialize < sink). Re-run with "
                "mutate=True to let the validator reorder, or fix the IR."
            ),
        ))
        return

    ir.stages = [ir.stages[i] for i in new_order]
    findings.append(Finding(
        severity=Severity.INFO,
        code="phase_reordered",
        detail=(
            "Reordered stages to honor coarse pipeline phases. "
            f"New order: [{', '.join(s.stage for s in ir.stages)}]."
        ),
    ))


# ----------------------------------------------------------------------------
# Shape compatibility / SegmentConcatenation auto-insert
# ----------------------------------------------------------------------------


_FAN_OUT_SHAPES = {OutputShape.FAN_OUT_SEGMENTS}


def _apply_shape_autoinsert(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
    inserted_out: list[StageRef],
) -> None:
    """Insert ``SegmentConcatenationStage`` between fan-out and whole-file consumers.

    Detection rule: walk the pipeline tracking the *current* output shape.
    When a stage requires :attr:`InputShape.WHOLE_FILE` but the upstream
    shape is in :data:`_FAN_OUT_SHAPES`, insert a concat stage so each
    source-file is reassembled. We never insert if a concat is already
    upstream, nor if the consumer tolerates :attr:`InputShape.ANY`.

    Reset rule: stages with ``output_shape=REBUILD_TASK`` or whose phase
    is ``concat`` reset the current shape to :attr:`OutputShape.PASSTHROUGH`.
    """

    if "SegmentConcatenationStage" not in reg:
        return  # registry doesn't have it; nothing we can do.

    def _shape_of(stage_name: str) -> tuple[InputShape, OutputShape]:
        entry = reg.get(stage_name)
        if entry is None:
            return InputShape.ANY, OutputShape.PASSTHROUGH
        return entry.card.input_shape, entry.card.output_shape

    i = 0
    current_shape: OutputShape = OutputShape.PASSTHROUGH
    while i < len(ir.stages):
        s = ir.stages[i]
        needs, emits = _shape_of(s.stage)
        if (
            needs == InputShape.WHOLE_FILE
            and current_shape in _FAN_OUT_SHAPES
        ):
            _force_upstream_vad_nested(ir.stages[:i])
            concat = StageRef(
                stage="SegmentConcatenationStage",
                params={},
                auto_inserted=True,
                insert_reason=(
                    f"restitch fan-out before {s.stage} (requires whole_file)"
                ),
            )
            ir.stages.insert(i, concat)
            inserted_out.append(concat)
            findings.append(Finding(
                severity=Severity.INFO,
                code="auto_inserted",
                detail=(
                    f"Auto-inserted SegmentConcatenationStage before {s.stage} "
                    f"because upstream shape was {current_shape.value!r} but "
                    f"{s.stage} requires whole_file. Set upstream "
                    "VADSegmentationStage.nested=true so concat receives "
                    "task.data['segments']."
                ),
                stage_index=i,
                stage_name="SegmentConcatenationStage",
            ))
            current_shape = OutputShape.PASSTHROUGH
            i += 2
            continue

        if emits == OutputShape.REBUILD_TASK:
            current_shape = OutputShape.PASSTHROUGH
        elif emits in _FAN_OUT_SHAPES:
            current_shape = emits
        elif s.stage == "SegmentConcatenationStage":
            current_shape = OutputShape.PASSTHROUGH
        i += 1


def _force_upstream_vad_nested(stages: list[StageRef]) -> None:
    """Switch the nearest upstream VAD to nested mode before inserting concat."""

    for ref in reversed(stages):
        if ref.stage == "VADSegmentationStage":
            ref.params = {**ref.params, "nested": True}
            return


# ----------------------------------------------------------------------------
# Terminal stage placement
# ----------------------------------------------------------------------------


def _check_terminal_placement(
    ir: PipelineIR,
    reg: CapabilityRegistry,
    findings: list[Finding],
) -> None:
    """A ``terminal: true`` stage must not be followed by audio-analytical work.

    Allowed followers: other ``terminal`` stages, sinks (WRITE_MANIFEST /
    AUDIO_TO_DOCUMENT), and :class:`Phase.MATERIALIZE` stages. Anything
    else (filters, inference, segmentation) running after a terminal
    stage is almost certainly a bug — its required keys have just been
    dropped from ``task.data``.
    """

    terminal_idx: int | None = None
    for i, s in enumerate(ir.stages):
        entry = reg.get(s.stage)
        if entry is None:
            continue
        if entry.card.terminal:
            terminal_idx = i
            break

    if terminal_idx is None:
        return

    allowed_phases = {Phase.MATERIALIZE, Phase.SINK, Phase.POLISH}
    for j in range(terminal_idx + 1, len(ir.stages)):
        s = ir.stages[j]
        entry = reg.get(s.stage)
        if entry is None:
            continue
        card = entry.card
        caps = set(card.capabilities) | set(card.also_handles)
        is_sink = bool(caps & {CapabilityTag.WRITE_MANIFEST, CapabilityTag.AUDIO_TO_DOCUMENT})
        if card.terminal or is_sink or card.phase in allowed_phases:
            continue
        findings.append(Finding(
            severity=Severity.ERROR,
            code="after_terminal_stage",
            detail=(
                f"{s.stage} runs after terminal stage "
                f"{ir.stages[terminal_idx].stage!r} (index {terminal_idx}). "
                "Terminal stages clear or rewrite task.data; analytical "
                "stages cannot consume their outputs."
            ),
            stage_index=j,
            stage_name=s.stage,
        ))


# ----------------------------------------------------------------------------
# Fingerprint
# ----------------------------------------------------------------------------


def _fingerprint(ir: PipelineIR) -> str:
    """Stable hash of the post-validation IR for cache / replay correlation."""

    import hashlib

    payload = ir.to_json(indent=0).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    "Finding",
    "Severity",
    "ValidationReport",
    "validate",
]
