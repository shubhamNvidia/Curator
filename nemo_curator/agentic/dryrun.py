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
"""Symbolic execution ("dry-run") of a :class:`PipelineIR`.

The dry-run walks the pipeline stage by stage and tracks an
:class:`AudioBatchState` — a structural model of what an actual
``AudioBatch`` / ``task`` object would look like at that point of the
flow. It uses ONLY the metadata declared on each :class:`StageCard`;
no audio is touched, no stage is constructed.

Why this exists
---------------

The validator (``nemo_curator.agentic.validator``) enforces hard
ordering rules (phase rank, topo, auto-insert). But "is the order
plausible end-to-end?" needs a forward simulation: does each stage's
input contract match the AudioBatch the previous stages would have
produced? Concrete examples:

- A stage requires ``input_shape=whole_file`` but the previous stage
  emits ``fan_out_segments`` — needs ``SegmentConcatenationStage``.
- A stage reads ``task.data["vad_segments"]`` but no upstream stage
  declares it as an output.
- A stage requires an in-memory waveform but no ``MonoConversionStage``
  has run yet.
- An analytical stage appears AFTER a terminal stage that cleared
  ``task.data``.

These are exactly the issues the staging critic needs to surface with
concrete pointers ("stage 4 ``SpeakerSeparationStage`` needs whole_file
but state is fan_out_segments"). The dry-run produces those pointers
deterministically — no LLM call, no hallucination risk.

Output
------

:func:`dry_run_pipeline` returns a :class:`DryRunReport` carrying:

- ``initial_state``: the simulated AudioBatch at the source.
- ``stages``: one :class:`DryRunStageTrace` per IR stage with input
  state, output state, phase / shape, and any ``issues`` strings
  detected at that step.
- Convenience accessors ``has_errors`` and ``all_issues``.

The trace also serves as a beautiful debug artifact — dump it to
``/tmp/dry_run.json`` and you can read exactly what the planner thought
each stage would do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_curator.agentic.cards import (
    InputShape,
    OutputShape,
    Phase,
    StageCard,
    StageCategory,
)
from nemo_curator.agentic.ir import PipelineIR
from nemo_curator.agentic.registry import CapabilityRegistry


# ----------------------------------------------------------------------------
# Initial state assumptions
# ----------------------------------------------------------------------------

# Top-level attributes that are present on EVERY ``AudioTask`` regardless
# of source — set by the base ``Task`` class. See
# ``nemo_curator/tasks/audio_task.py``. Fields like ``_audio_id`` or
# ``source_id`` do NOT exist on AudioTask; lineage is implicit via the
# ``task_id`` suffix (e.g. ``_seg_0``) and via timing fields inside
# ``task.data``.
_INITIAL_TOP_LEVEL: frozenset[str] = frozenset({"task_id", "dataset_name", "data"})

# task.data keys that exist on a fresh AudioTask coming out of any
# manifest / directory reader. Readers also propagate arbitrary manifest
# row fields (``text``, ``language``, etc.) into ``task.data``, but we
# only model the one guaranteed key here so downstream cards can be
# explicit about what they additionally rely on.
_INITIAL_DATA: frozenset[str] = frozenset({"audio_filepath"})


# ----------------------------------------------------------------------------
# State + trace dataclasses
# ----------------------------------------------------------------------------


@dataclass
class AudioBatchState:
    """Symbolic snapshot of an ``AudioBatch`` / ``task`` state.

    The dry-run carries one of these between stages. Every field
    corresponds to a precondition / postcondition that some
    :class:`StageCard` may care about.
    """

    top_level_keys: set[str]
    data_keys: set[str]
    # "whole_file" | "fan_out_segments" | "fan_out_speakers" |
    # "nested_segments". The post-concat shape is "whole_file".
    shape: str
    audio_in_memory: bool
    audio_on_disk: bool
    sample_rate: int | None
    is_mono: bool | None  # True | False | None=unknown
    # True after a ``terminal: true`` stage has cleared ``task.data``;
    # only sink / materialize stages may run while frozen.
    frozen: bool

    def snapshot(self) -> AudioBatchState:
        return AudioBatchState(
            top_level_keys=set(self.top_level_keys),
            data_keys=set(self.data_keys),
            shape=self.shape,
            audio_in_memory=self.audio_in_memory,
            audio_on_disk=self.audio_on_disk,
            sample_rate=self.sample_rate,
            is_mono=self.is_mono,
            frozen=self.frozen,
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "data_keys": sorted(self.data_keys),
            "shape": self.shape,
            "audio_in_memory": self.audio_in_memory,
            "audio_on_disk": self.audio_on_disk,
            "sample_rate": self.sample_rate,
            "is_mono": self.is_mono,
            "frozen": self.frozen,
        }


@dataclass
class DryRunStageTrace:
    """Per-stage trace entry."""

    stage_index: int
    stage_name: str
    phase: str
    input_shape: str
    output_shape: str
    input_state: AudioBatchState
    output_state: AudioBatchState
    issues: list[str]


@dataclass
class DryRunReport:
    """Full dry-run output: initial state + per-stage trace."""

    initial_state: AudioBatchState
    stages: list[DryRunStageTrace]

    @property
    def has_errors(self) -> bool:
        return any(t.issues for t in self.stages)

    @property
    def all_issues(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in self.stages:
            for msg in t.issues:
                out.append({
                    "stage_index": t.stage_index,
                    "stage_name": t.stage_name,
                    "issue": msg,
                })
        return out

    def to_payload(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_summary(),
            "trace": [
                {
                    "stage_index": t.stage_index,
                    "stage_name": t.stage_name,
                    "phase": t.phase,
                    "input_shape": t.input_shape,
                    "output_shape": t.output_shape,
                    "input_state": t.input_state.to_summary(),
                    "output_state": t.output_state.to_summary(),
                    "issues": t.issues,
                }
                for t in self.stages
            ],
            "has_errors": self.has_errors,
        }


# ----------------------------------------------------------------------------
# Shape + state transition helpers
# ----------------------------------------------------------------------------


def _shape_compatible(needed: InputShape, current: str) -> bool:
    """True if a stage's required ``input_shape`` is satisfied by ``current``.

    Note on ``fan_out_speakers`` → ``whole_file``: this is treated as
    compatible.
    :class:`~nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage`
    produces one audio per task (one per detected speaker), and each
    per-speaker waveform spans the full source duration with silence
    padding outside that speaker's regions (see the reference layout in
    ``advanced_pipelines/audio_data_filter._build_full_pipeline``). From
    any downstream stage's per-task perspective the task therefore looks
    identical to one produced by a plain ``whole_file → whole_file``
    step, with timestamps already in source-file coordinates.
    ``fan_out_segments`` is intentionally *not* relaxed because those
    tasks share a parent file: re-stitching via ``SegmentConcatenationStage``
    is meaningful there and the validator auto-inserts it.
    """

    if needed == InputShape.ANY:
        return True
    if needed == InputShape.WHOLE_FILE:
        return current in {"whole_file", "fan_out_speakers"}
    if needed == InputShape.NESTED_SEGMENTS:
        return current in {"nested_segments", "whole_file"}
    if needed == InputShape.FANNED_OUT:
        return current.startswith("fan_out")
    return True


def _shape_after(card: StageCard, current: str, params: dict[str, Any] | None = None) -> str:
    # VAD has TWO behaviours selected at runtime by the `nested` flag.
    # The card declares the default (`fan_out_segments`); switch to
    # `nested_segments` if the planner picked `nested=True` so the
    # downstream `SegmentConcatenationStage` is recognised as the
    # legal continuation.
    if (
        card.name == "VADSegmentationStage"
        and params is not None
        and bool(params.get("nested"))
    ):
        return "nested_segments"

    out = card.output_shape
    if out == OutputShape.PASSTHROUGH:
        return current
    if out == OutputShape.FAN_OUT_SEGMENTS:
        return "fan_out_segments"
    if out == OutputShape.FAN_OUT_SPEAKERS:
        return "fan_out_speakers"
    if out == OutputShape.NESTED_SEGMENTS:
        return "nested_segments"
    if out == OutputShape.FAN_IN:
        return "whole_file"
    if out == OutputShape.REBUILD_TASK:
        return "whole_file"
    if out == OutputShape.WRITE_FILES:
        return current
    return current


def _is_sink_or_materialize(card: StageCard) -> bool:
    """Stages that may run after a terminal step has frozen task.data."""

    return card.category == StageCategory.SINK or card.phase in {
        Phase.SINK,
        Phase.MATERIALIZE,
    }


def _params_implied_sample_rate(stage_name: str, params: dict[str, Any]) -> int | None:
    """Pull SR out of MonoConversion / Resample params, if set."""

    if stage_name == "MonoConversionStage":
        v = params.get("output_sample_rate")
        return int(v) if v is not None else None
    if stage_name == "ResampleAudioStage":
        v = params.get("target_sample_rate")
        return int(v) if v is not None else None
    return None


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def initial_state() -> AudioBatchState:
    """Fresh AudioBatch state at the head of any pipeline."""

    return AudioBatchState(
        top_level_keys=set(_INITIAL_TOP_LEVEL),
        data_keys=set(_INITIAL_DATA),
        shape="whole_file",
        audio_in_memory=False,
        audio_on_disk=True,
        sample_rate=None,
        is_mono=None,
        frozen=False,
    )


def dry_run_pipeline(ir: PipelineIR, registry: CapabilityRegistry) -> DryRunReport:
    """Symbolically execute ``ir`` and return a trace + issues report."""

    state = initial_state()
    initial = state.snapshot()
    trace: list[DryRunStageTrace] = []

    for i, sref in enumerate(ir.stages):
        entry = registry.get(sref.stage)
        if entry is None:
            trace.append(DryRunStageTrace(
                stage_index=i,
                stage_name=sref.stage,
                phase="unknown",
                input_shape="unknown",
                output_shape="unknown",
                input_state=state.snapshot(),
                output_state=state.snapshot(),
                issues=[f"unknown stage {sref.stage!r} — not present in registry"],
            ))
            continue

        card = entry.card
        input_snapshot = state.snapshot()
        issues: list[str] = []

        # --- Preconditions ----------------------------------------------------

        # Shape compatibility
        if not _shape_compatible(card.input_shape, state.shape):
            need = card.input_shape.value
            issues.append(
                f"input_shape={need!r} required but AudioBatch shape is "
                f"{state.shape!r}; insert SegmentConcatenationStage upstream"
            )

        # Required top-level attributes
        for k in card.inputs.top_level:
            if k not in state.top_level_keys:
                issues.append(f"missing top-level attribute task.{k} on the AudioBatch")

        # Required task.data keys (source-phase stages create their own
        # initial keys, so we skip the check there)
        if card.phase != Phase.SOURCE:
            for k in card.inputs.data:
                if k not in state.data_keys:
                    issues.append(
                        f"reads task.data[{k!r}] but no upstream stage declares "
                        f"it as an output"
                    )

        # Sample-rate precondition
        if card.requires_sample_rate is not None:
            if state.sample_rate is None:
                issues.append(
                    f"requires sample_rate={card.requires_sample_rate} but the upstream "
                    f"sample rate is not yet established — insert MonoConversionStage "
                    f"or ResampleAudioStage upstream"
                )
            elif state.sample_rate != card.requires_sample_rate:
                issues.append(
                    f"requires sample_rate={card.requires_sample_rate} but upstream "
                    f"provides {state.sample_rate}"
                )

        # Mono precondition
        if card.requires_mono and state.is_mono is False:
            issues.append("requires mono audio but the upstream signal is stereo")

        # In-memory waveform precondition
        if card.requires_in_memory_waveform and not state.audio_in_memory:
            issues.append(
                "requires an in-memory waveform but no upstream stage has loaded "
                "audio into memory (insert MonoConversionStage / ResampleAudioStage)"
            )

        # On-disk path precondition
        if card.requires_on_disk_path and not state.audio_on_disk:
            issues.append(
                "requires an on-disk audio path but upstream stages have "
                "loaded audio only into memory"
            )

        # Frozen / terminal-then-analytical
        if state.frozen and not _is_sink_or_materialize(card):
            issues.append(
                "runs AFTER a terminal stage (task.data was cleared) — only "
                "sink / materialize stages are allowed past a terminal stage"
            )

        # --- Apply effects ---------------------------------------------------
        # We apply effects regardless of preconditions failing so the trace
        # keeps going and the report can surface every issue at once.
        for k in card.outputs.top_level:
            state.top_level_keys.add(k)
        for k in card.outputs.data:
            state.data_keys.add(k)
        if (
            card.name == "VADSegmentationStage"
            and bool((sref.params or {}).get("nested"))
            and card.nested_segment_key
        ):
            state.data_keys.add(card.nested_segment_key)
        for k in card.produces_keys_after_run:
            state.data_keys.add(k)
        for k in card.drops_keys_after_run:
            state.data_keys.discard(k)

        state.shape = _shape_after(card, state.shape, sref.params or {})

        if card.produces_in_memory_waveform:
            state.audio_in_memory = True
        if card.produces_on_disk_files:
            state.audio_on_disk = True

        sr_set = _params_implied_sample_rate(card.name, sref.params or {})
        if sr_set is not None:
            state.sample_rate = sr_set
        if card.name == "MonoConversionStage":
            state.is_mono = True

        if card.terminal:
            state.frozen = True

        trace.append(DryRunStageTrace(
            stage_index=i,
            stage_name=card.name,
            phase=card.phase.value,
            input_shape=card.input_shape.value,
            output_shape=card.output_shape.value,
            input_state=input_snapshot,
            output_state=state.snapshot(),
            issues=issues,
        ))

    return DryRunReport(initial_state=initial, stages=trace)


__all__ = [
    "AudioBatchState",
    "DryRunReport",
    "DryRunStageTrace",
    "dry_run_pipeline",
    "initial_state",
]
