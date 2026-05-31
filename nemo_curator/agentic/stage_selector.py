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
"""Stage-application engine — IntentCategories → ordered StageRefs.

This module is the deterministic compiler heart. It walks the V2
:class:`IntentCategories` ingredient list and emits the IR's stage
sequence directly. There is no LLM call here; every decision is the
result of a row in the §4 stage-application matrix in
``INTENT_V2.md``.

The selector returns stages in a *partial* order (read sequence). The
validator and the dry-run pass are still responsible for topological
sort, auto-insertion of normalizers, and key-flow checks. We deliberately
keep the selector small so the matrix can be reviewed at a glance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_curator.agentic.cards import StageCard
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    QualityGate,
    SigmosAxis,
)
from nemo_curator.agentic.ir import SinkSpec, StageRef
from nemo_curator.agentic.registry import CapabilityRegistry


# ----------------------------------------------------------------------------
# PreserveByValueStage operator semantics
# ----------------------------------------------------------------------------
#
# ``PreserveByValueStage`` is a generic key/value filter — it KEEPS the task
# when ``task.data[input_value_key] <operator> target_value`` evaluates True
# and drops it otherwise. The agentic layer used to wire only a couple of
# operators, so user prompts of the form "keep at least N speakers" /
# "duration greater than 60s" silently fell on the floor. The full
# operator table the selector now exposes:
#
#   eq  — exact match.        Use for "exactly N".                target_count
#   ne  — not equal.          Use for "anything but N".            (reserved)
#   le  — keep at_most N.     Use for "≤ X" / "no more than X".    max_count, wer_max, duration_max_sec
#   lt  — keep strictly < N.  Use for "< X" / "less than X".       (callers can pass directly)
#   ge  — keep at_least N.    Use for "≥ X" / "at least X".        min_count, duration_min_sec (for original_files)
#   gt  — keep strictly > N.  Use for "> X" / "greater than X".    (callers can pass directly)
#
# Every emission goes through :func:`_preserve_by_value` so the operator
# choice is centralized and self-documenting. Adding a new gate is just
# "another helper call with the right operator string" — no copy-paste.


_PRESERVE_BY_VALUE_OPERATORS: frozenset[str] = frozenset({"lt", "le", "eq", "ne", "ge", "gt"})


def _preserve_by_value(
    *,
    key: str,
    operator: str,
    value: int | float | str,
    reason: str,
) -> StageRef:
    """Build a single ``PreserveByValueStage`` with the chosen comparison.

    ``operator`` must be one of ``{lt, le, eq, ne, ge, gt}``. The stage
    KEEPS tasks satisfying ``task.data[key] <operator> value``.

    ``reason`` is stamped onto :attr:`StageRef.insert_reason` so it shows
    up in the validator findings / the web UI — useful for debugging
    "why did this row get dropped?".
    """

    if operator not in _PRESERVE_BY_VALUE_OPERATORS:
        msg = (
            f"PreserveByValueStage operator {operator!r} not in "
            f"{sorted(_PRESERVE_BY_VALUE_OPERATORS)}"
        )
        raise ValueError(msg)
    return StageRef(
        stage="PreserveByValueStage",
        params={
            "input_value_key": key,
            "target_value": value,
            "operator": operator,
        },
        auto_inserted=True,
        insert_reason=reason,
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def select_stages(
    intent: IntentCategories,
    *,
    registry: CapabilityRegistry,
    sink: SinkSpec,
) -> list[StageRef]:
    """Compile an intent into an ordered list of :class:`StageRef`.

    The returned sequence does NOT include the source reader or the
    manifest writer — those are owned by the planner because they need
    the ``SourceSpec`` (and the validator auto-inserts the writer when
    missing). Everything in between, in pipeline order, is here.

    Pipeline shape rules (from the §4 matrix in ``INTENT_V2.md``):

    - **Cleaning flow** (``output_unit=='original_files'`` and the user
      asked for VAD via ``segmentation.speech_policy``): emit
      ``VAD(nested=True) → quality_filters → SegmentConcatenationStage``
      as a bound block, then everything else operates on the *cleaned*
      whole-file waveform. Nested-aware filters (UTMOS / SIGMOS / Band)
      drop bad internal segments before concat re-stitches the survivors.
    - **Output flow** (``output_unit`` is a segment unit such as
      ``speech_segments`` / ``single_speaker_clips`` / ``long_windows``):
      the primary segmenter runs first (``VAD`` for ``speech_segments``,
      ``SpeakerSeparation`` from :func:`_speaker_stages` for
      ``single_speaker_clips``, ``SplitLongAudio`` for ``long_windows``);
      quality filters / annotations apply per-segment afterwards;
      ``SegmentExtractionStage`` writes one file per surviving segment.
    - **Post-segmenter VAD trim** (the second half of the user's
      "VAD AFTER segmentation" rule): when the primary segmenter is
      ``SpeakerSeparation`` (or ``SplitLongAudio``) and the user gave us
      ``duration_min_sec`` / ``duration_max_sec`` or a non-OFF
      ``speech_policy``, drop a ``VAD(nested=False)`` *after* the
      segmenter. ``SpeakerSeparation`` returns per-speaker waveforms that
      span the full source duration (silence outside the speaker's
      regions) — see
      ``advanced_pipelines/audio_data_filter._build_full_pipeline`` for
      the reference layout — so VAD's per-task ``start_ms`` / ``end_ms``
      already live in source-file coordinates and
      ``SegmentExtractionStage`` can read them directly. See
      :func:`_post_segmenter_vad_trim`.
    """

    cleaning = _is_cleaning_flow(intent)

    stages: list[StageRef] = []
    stages.extend(_output_normalizers(intent, sink))
    # File-level speaker filtering / annotation has to run BEFORE any
    # segmenter that fans out into per-segment rows. Sortformer reports
    # the number of speakers it hears *in the row it's given*, so a
    # 2-30 s VAD clip will almost always look monospeaker even when the
    # source file has many. Keeping Sortformer+PBV upstream of VAD means
    # the user's "single-speaker files only" filter actually drops the
    # right files. SPLIT goes in its usual place (it's the segmenter).
    stages.extend(_file_level_speaker_stages(intent))
    stages.extend(_segmentation_stages(intent, sink, cleaning_flow=cleaning))
    stages.extend(_quality_stages(intent))
    stages.extend(_cleaning_concat_stages(cleaning_flow=cleaning))
    stages.extend(_speaker_stages(intent))
    stages.extend(_post_segmenter_vad_trim(intent))
    stages.extend(_text_stages(intent))
    stages.extend(_alm_stages(intent))
    stages.extend(_whole_file_duration_filter_stages(intent))
    stages.extend(_segment_extraction_stages(intent, sink))

    return [
        ref.model_copy(update={"params": _coerce_params(ref.stage, ref.params, registry)})
        for ref in _dedupe_by_identity(stages)
    ]


def _is_cleaning_flow(intent: IntentCategories) -> bool:
    """True when the user wants *internal* audio cleaning before further work.

    Signal: the user kept whole-file outputs (``output_unit=='original_files'``)
    AND explicitly turned on a speech-aware policy
    (``segmentation.speech_policy`` is ANNOTATE or FILTER). That is the
    only signal in :class:`IntentCategories` today that says "split for
    cleaning, then re-stitch into one waveform". Pure annotation flows
    (e.g. UTMOS-annotate without speech_policy) stay whole-file with no
    VAD, matching the working baseline.
    """

    seg = intent.segmentation
    if seg.output_unit != "original_files":
        return False
    return seg.speech_policy in {FilterMode.ANNOTATE, FilterMode.FILTER}


def _cleaning_concat_stages(*, cleaning_flow: bool) -> list[StageRef]:
    """Emit the trailing :class:`SegmentConcatenationStage` for the
    cleaning flow so the cleaning block (VAD(nested=True) → quality →
    Concat) is bound together by the selector instead of being repaired
    by the validator later."""

    if not cleaning_flow:
        return []
    return [StageRef(
        stage="SegmentConcatenationStage",
        params={},
        auto_inserted=True,
        insert_reason=(
            "cleaning flow: re-stitch surviving nested-VAD segments after "
            "quality filtering, before downstream whole-file consumers"
        ),
    )]


# ----------------------------------------------------------------------------
# Output normalization
# ----------------------------------------------------------------------------


def _output_normalizers(intent: IntentCategories, sink: SinkSpec) -> list[StageRef]:
    """Resample/Mono pair driven by the user's output choices.

    Rules from §4:

    - ``sample_rate is int`` and ``resample_input=True``  → Resample → Mono.
    - ``sample_rate is int`` and ``resample_input=False`` → Mono (strict).
    - ``channels=mono`` only                              → Mono at 48 kHz.

    Container-format policy:
        When the pipeline ends in ``SegmentExtractionStage`` we keep the
        resampled intermediates in WAV (lossless, fast to decode) and let
        the final stage re-encode each clip into the user's chosen
        ``sink.output_format``. For whole-file flows (no extraction) we
        write the user's chosen format directly because there is no later
        stage to do the conversion.
    """

    out: list[StageRef] = []
    sr = intent.output.sample_rate
    want_mono = intent.output.channels == "mono"
    concrete_sr = isinstance(sr, int)
    has_extraction_stage = intent.segmentation.output_unit in {
        "speech_segments",
        "single_speaker_clips",
    }
    intermediate_format = "wav" if has_extraction_stage else sink.output_format

    if concrete_sr and intent.output.resample_input is not False:
        # ``resample_input`` is True or unset (default behavior = convert).
        out.append(StageRef(
            stage="ResampleAudioStage",
            params={
                "resampled_audio_dir": str(Path(sink.target_dir) / "_resampled"),
                "target_sample_rate": int(sr),  # type: ignore[arg-type]
                "target_nchannels": 1 if want_mono else None,
                "target_format": intermediate_format,
                "resampled_audio_filepath_key": "audio_filepath",
            },
            auto_inserted=True,
            insert_reason=(
                "user picked output.sample_rate with resample_input != False"
                + ("; intermediate kept as wav (final format applied in SegmentExtraction)"
                   if has_extraction_stage and intermediate_format == "wav"
                     and sink.output_format != "wav"
                   else "")
            ),
        ))

    needs_strict_mono = (
        concrete_sr and intent.output.resample_input is False
    ) or (want_mono and not concrete_sr)

    if needs_strict_mono or concrete_sr:
        target_sr = int(sr) if concrete_sr else 48000  # type: ignore[arg-type]
        out.append(StageRef(
            stage="MonoConversionStage",
            params={
                "output_sample_rate": target_sr,
                "strict_sample_rate": intent.output.resample_input is False,
            },
            auto_inserted=True,
            insert_reason=(
                "user requested mono / strict sample-rate; load into memory at target SR"
            ),
        ))

    return out


# ----------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------


def _segmentation_stages(
    intent: IntentCategories,
    sink: SinkSpec,  # noqa: ARG001 — retained for symmetry with siblings
    *,
    cleaning_flow: bool,
) -> list[StageRef]:
    """Primary-segmenter stage per the §4 segmentation rows.

    Selection rule (one segmenter per pipeline):

    - ``speech_segments``:  ``VAD(nested=False)`` is the segmenter — it
      fans out into speech-only clips that downstream filters consume.
    - ``single_speaker_clips``:  no upfront VAD. ``SpeakerSeparationStage``
      from :func:`_speaker_stages` is the segmenter; quality filters /
      annotations apply per-speaker after it fans out. Adding VAD here
      would mean "split for cleaning, then split again per speaker" —
      that's the cleaning flow, which only applies when
      ``output_unit=='original_files'``.
    - ``long_windows``:  ``SplitLongAudioStage`` is the segmenter.
    - ``original_files`` + cleaning flow:  ``VAD(nested=True)`` opens the
      bound cleaning block (``VAD → quality_filters → Concat``). The
      trailing concat is emitted by :func:`_cleaning_concat_stages`.
    - ``original_files`` without cleaning:  no VAD; the pipeline stays
      whole-file end to end.
    """

    seg = intent.segmentation
    out: list[StageRef] = []

    if seg.output_unit == "speech_segments":
        out.append(_vad_stage(intent, nested=False))
    elif seg.output_unit == "single_speaker_clips":
        # No upfront VAD — SpeakerSep is the segmenter. (See module docstring.)
        pass
    elif seg.output_unit == "long_windows":
        window = seg.long_window_sec if seg.long_window_sec is not None else 120.0
        out.append(StageRef(
            stage="SplitLongAudioStage",
            params={"window_sec": float(window)},
        ))
    elif seg.output_unit == "original_files" and cleaning_flow:
        out.append(_vad_stage(intent, nested=True, force_min_duration=0.5))

    return out


def _vad_stage(intent: IntentCategories, *, nested: bool, force_min_duration: float | None = None) -> StageRef:
    seg = intent.segmentation
    params: dict[str, Any] = {"nested": nested}
    if seg.duration_min_sec is not None:
        params["min_duration_sec"] = float(seg.duration_min_sec)
    elif force_min_duration is not None:
        params["min_duration_sec"] = float(force_min_duration)
    if seg.duration_max_sec is not None:
        params["max_duration_sec"] = float(seg.duration_max_sec)
    if seg.vad_threshold is not None:
        params["threshold"] = float(seg.vad_threshold)
    if seg.speech_pad_ms is not None:
        params["speech_pad_ms"] = int(seg.speech_pad_ms)
    return StageRef(stage="VADSegmentationStage", params=params)


def _post_segmenter_vad_trim(intent: IntentCategories) -> list[StageRef]:
    """Emit a ``VAD(nested=False)`` AFTER the primary segmenter.

    This is the second half of the user's "VAD AFTER segmentation" rule:
    when a non-VAD segmenter created the output units (``SpeakerSeparation``
    for ``single_speaker_clips``, ``SplitLongAudio`` for ``long_windows``)
    AND the user gave us a real reason to trim per-segment — namely

    - a per-clip duration cap (``duration_min_sec`` / ``duration_max_sec``), OR
    - a non-OFF ``speech_policy`` (drop / annotate non-speech inside each
      already-fanned-out clip)

    we run VAD as a post-segmenter trim. The result is fan-out segments,
    each ≤ ``max_duration_sec`` and (when ``speech_policy != OFF``)
    speech-only. ``SegmentExtractionStage`` then writes one file per
    survivor.

    Not emitted when:

    - ``output_unit == 'speech_segments'`` — VAD is already the segmenter
      upstream; running it again would double-split.
    - ``output_unit == 'original_files'`` — that flow is handled by the
      cleaning block (``VAD(nested=True) → quality → Concat``) instead.
    - Neither a duration cap nor a speech policy is set — there would be
      nothing for VAD to enforce.

    NB: the coordinate alignment works because
    :class:`~nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage`
    returns per-speaker waveforms padded with silence to the full source
    duration. VAD's per-task ``start_ms`` / ``end_ms`` are therefore
    already in the source-file coordinate system that
    :class:`~nemo_curator.stages.audio.io.extract_segments.SegmentExtractionStage`
    reads from when it opens ``original_file``.
    """

    seg = intent.segmentation
    if seg.output_unit not in {"single_speaker_clips", "long_windows"}:
        return []
    has_duration = (
        seg.duration_min_sec is not None or seg.duration_max_sec is not None
    )
    has_speech_policy = seg.speech_policy != FilterMode.OFF
    if not (has_duration or has_speech_policy):
        return []
    # When the user only specified a duration cap (no speech_policy), don't
    # let VAD's default ``min_duration_sec=2.0`` accidentally drop short
    # but legitimate clips that the user didn't ask to filter.
    force_min = None if has_speech_policy or seg.duration_min_sec is not None else 0.0
    return [_vad_stage(intent, nested=False, force_min_duration=force_min)]


def _segment_extraction_stages(intent: IntentCategories, sink: SinkSpec) -> list[StageRef]:
    unit = intent.segmentation.output_unit
    if unit not in {"speech_segments", "single_speaker_clips"}:
        return []
    out_dir = str(Path(sink.target_dir) / sink.audio_subdir)
    return [
        StageRef(
            stage="TimestampMapperStage",
            params={},
        ),
        StageRef(
            stage="SegmentExtractionStage",
            params={
                "output_dir": out_dir,
                "output_format": sink.output_format,
            },
        ),
    ]


# ----------------------------------------------------------------------------
# Quality (mos / sigmos / band) — pure-annotate + PreserveByValueStage drops
# ----------------------------------------------------------------------------
#
# Contract (the "score → gate" split):
#
# 1. ``UTMOSFilterStage`` and ``SIGMOSFilterStage`` are ALWAYS emitted with
#    every threshold pinned to ``0.0``. Both stages still write their score
#    keys to ``task.data`` regardless of threshold, so a 0 floor means
#    "annotate every row, drop nothing".
# 2. The actual KEEP/DROP decision is made by ``PreserveByValueStage``,
#    which has the full operator matrix ``{lt, le, eq, ne, ge, gt}``. The
#    selector compiles intent fields into one or more PBV rows:
#
#       quality.mos_threshold=X (mos=FILTER)        -> PBV(utmos_mos, ge, X)
#       quality.sigmos_thresholds[axis]=X (FILTER)  -> PBV(sigmos_<axis>, ge, X)
#       quality.band_value=V (band=FILTER)          -> PBV(band_prediction, eq, V)
#                                                       (BandFilterStage is
#                                                        also emitted as the
#                                                        upstream score
#                                                        producer because it
#                                                        has no annotate-only
#                                                        knob today.)
#       quality.gates[...]                          -> PBV(<key>, <op>, <val>)
#
# 3. Free-form ``quality.gates`` always run LAST and can target ANY
#    operator — this is how "drop MOS > 4" / "noise < 2" / "narrow-band
#    only" / etc. become reachable without touching the schema.
#
# Why this matters:
#   * Single source of truth for drop logic ➜ no more "stage drops, then
#     PBV drops again" double-jeopardy.
#   * The operator semantics declared in the cards (every MOS axis is
#     "higher = better" on the 0-5 ACR scale) get expressed once, in PBV,
#     instead of being baked into each scoring stage.
#   * Card aggressiveness presets keep working transparently: the smart
#     clarifier still emits ``mos_threshold`` / ``sigmos_thresholds`` /
#     ``sigmos_axes``; the selector now compiles them into PBV rows.


# Map a SIGMOS axis short name to the task.data key the stage actually
# writes. Stays local because it's only used by the selector for the
# legacy-field → PBV translation.
_SIGMOS_AXIS_KEYS: dict[SigmosAxis, str] = {
    "ovrl": "sigmos_ovrl",
    "noise": "sigmos_noise",
    "sig": "sigmos_sig",
    "col": "sigmos_col",
    "disc": "sigmos_disc",
    "loud": "sigmos_loud",
    "reverb": "sigmos_reverb",
}


def _utmos_required(intent: IntentCategories) -> bool:
    """True iff ANY downstream consumer needs the ``utmos_mos`` key."""

    q = intent.quality
    if q.mos != FilterMode.OFF:
        return True
    return any(g.key == "utmos_mos" for g in q.gates)


def _sigmos_required_axes(intent: IntentCategories) -> list[SigmosAxis]:
    """Which SIGMOS axes need to be present in ``task.data``.

    Combines the legacy ``sigmos_axes`` / ``sigmos_thresholds`` keys with
    anything referenced in ``gates``. Returns axes in a deterministic
    order for snapshot stability.
    """

    q = intent.quality
    requested: set[SigmosAxis] = set()
    if q.sigmos != FilterMode.OFF:
        requested.update(q.sigmos_axes)
        requested.update(q.sigmos_thresholds.keys())
    for g in q.gates:
        if not g.key.startswith("sigmos_"):
            continue
        axis = g.key.removeprefix("sigmos_")
        if axis in _SIGMOS_AXIS_KEYS:
            requested.add(axis)  # type: ignore[arg-type]
    if q.sigmos != FilterMode.OFF and not requested:
        # Default axes from the project's canonical AudioDataFilter
        # combo (overall quality + background noise are the most
        # universally useful pair).
        requested.update(["ovrl", "noise"])
    # Stable order: ovrl, noise, sig, col, disc, loud, reverb.
    return [axis for axis in _SIGMOS_AXIS_KEYS if axis in requested]


def _quality_stages(intent: IntentCategories) -> list[StageRef]:
    q = intent.quality
    out: list[StageRef] = []

    # --- Phase 1: pure annotators (threshold = 0 across the board). ------
    if _utmos_required(intent):
        out.append(StageRef(
            stage="UTMOSFilterStage",
            params={"mos_threshold": 0.0},
            auto_inserted=True,
            insert_reason=(
                "UTMOSFilterStage pinned to mos_threshold=0.0 — drop "
                "decisions are made downstream by PreserveByValueStage."
            ),
        ))

    sigmos_axes = _sigmos_required_axes(intent)
    if sigmos_axes:
        out.append(StageRef(
            stage="SIGMOSFilterStage",
            params={f"{axis}_threshold": 0.0 for axis in sigmos_axes},
            auto_inserted=True,
            insert_reason=(
                "SIGMOSFilterStage pinned to threshold=0.0 on axes "
                f"{sigmos_axes} — drop decisions are made downstream by "
                "PreserveByValueStage."
            ),
        ))

    # BandFilterStage has no annotate-only knob; emit it as-is when the
    # user explicitly opted into bandwidth filtering. The PBV mirror
    # below makes the drop decision visible at the IR level too.
    if q.band == FilterMode.FILTER and q.band_value is not None:
        out.append(StageRef(
            stage="BandFilterStage",
            params={"band_value": q.band_value},
        ))

    # --- Phase 2: PreserveByValueStage rows for actual drops. -----------
    # 2a) Legacy UTMOS floor.
    if q.mos == FilterMode.FILTER:
        threshold = float(q.mos_threshold if q.mos_threshold is not None else 3.4)
        out.append(_preserve_by_value(
            key="utmos_mos",
            operator="ge",
            value=threshold,
            reason=(
                f"quality.mos=FILTER, mos_threshold={threshold} → keep "
                f"rows where utmos_mos >= {threshold}."
            ),
        ))

    # 2b) Legacy SIGMOS per-axis floors.
    if q.sigmos == FilterMode.FILTER:
        filter_axes = q.sigmos_axes or ["ovrl", "noise"]
        for axis in filter_axes:
            value = float(q.sigmos_thresholds.get(axis, 3.5))
            out.append(_preserve_by_value(
                key=_SIGMOS_AXIS_KEYS[axis],
                operator="ge",
                value=value,
                reason=(
                    f"quality.sigmos=FILTER, axis {axis} threshold={value} "
                    f"→ keep rows where {_SIGMOS_AXIS_KEYS[axis]} >= {value}."
                ),
            ))

    # 2c) Legacy bandwidth class as PBV.
    if q.band == FilterMode.FILTER and q.band_value is not None:
        out.append(_preserve_by_value(
            key="band_prediction",
            operator="eq",
            value=str(q.band_value),
            reason=(
                f"quality.band=FILTER, band_value={q.band_value!r} → keep "
                f"rows where band_prediction == {q.band_value!r}."
            ),
        ))

    # 2d) Free-form user gates — appended last so they trump legacy fields
    # when the user explicitly asked for a non-``ge`` comparison.
    for gate in q.gates:
        out.append(_quality_gate_to_pbv(gate))

    return out


def _quality_gate_to_pbv(gate: QualityGate) -> StageRef:
    """Compile a :class:`QualityGate` into a ``PreserveByValueStage`` ref."""

    if gate.key == "band_prediction":
        value: int | float | str = str(gate.value)
    else:
        try:
            value = float(gate.value)  # MOS axes are all numeric.
        except (TypeError, ValueError) as exc:
            msg = (
                f"QualityGate for key={gate.key!r} expects a numeric value; "
                f"got {gate.value!r} ({type(gate.value).__name__})."
            )
            raise ValueError(msg) from exc
    return _preserve_by_value(
        key=gate.key,
        operator=gate.operator,
        value=value,
        reason=(
            f"quality.gates entry → keep rows where {gate.key} "
            f"{gate.operator} {gate.value!r}."
        ),
    )


# ----------------------------------------------------------------------------
# Speakers
# ----------------------------------------------------------------------------


def _file_level_speaker_stages(intent: IntentCategories) -> list[StageRef]:
    """Emit file-level diarization + speaker filtering BEFORE segmentation.

    ``InferenceSortformerStage`` accepts any input shape and writes
    ``num_speakers`` on the row it processes. If the pipeline first
    fans out into VAD segments and *then* runs Sortformer, each
    per-segment row reports the speakers *in that segment* (usually
    one, because the clip is short) — so a downstream
    ``PreserveByValueStage(num_speakers, op, value)`` would never drop
    the files the user actually meant to filter.

    We therefore emit Sortformer + the speaker-count PBV(s) *before*
    any segmenter when:

    * ``speakers.mode == ANNOTATE`` — the user wants the per-file
      speaker count carried through to every downstream row, and
    * ``speakers.mode == FILTER`` — the speaker-count gate is
      genuinely a file-level decision (drop the file if N != target).

    ``SPLIT`` stays in :func:`_speaker_stages` because there
    ``SpeakerSeparationStage`` *is* the segmenter and must run later.
    """

    spk = intent.speakers
    if spk.mode not in (FilterMode.ANNOTATE, FilterMode.FILTER):
        return []

    # ``phase_override='preprocess'`` is the key: the validator's
    # phase-aware reorder would otherwise push Sortformer (ANALYZE)
    # after VAD (SEGMENT) and the file-level filter would silently
    # become a per-segment no-op.
    out: list[StageRef] = [StageRef(
        stage="InferenceSortformerStage",
        params={},
        auto_inserted=True,
        phase_override="preprocess",
        insert_reason=(
            f"speakers.mode={spk.mode.value} → file-level diarization "
            "must run before any segmenter so num_speakers reflects the "
            "whole source file, not a single VAD segment."
        ),
    )]

    if spk.mode != FilterMode.FILTER:
        return out

    # FILTER → translate the count bounds into PBV gate(s). Same
    # operator ladder as the post-segmenter variant used to use.
    # PBVs also need the preprocess override to stay co-located with
    # the upstream Sortformer; otherwise phase-sort would push them
    # past the segmenter, where they'd run on per-segment rows.
    def _file_level_pbv(*, key, operator, value, reason):
        ref = _preserve_by_value(key=key, operator=operator, value=value, reason=reason)
        ref.phase_override = "preprocess"
        return ref

    if spk.target_count is not None:
        out.append(_file_level_pbv(
            key="num_speakers",
            operator="eq",
            value=int(spk.target_count),
            reason=(
                f"speakers.target_count={spk.target_count} → keep "
                f"files where num_speakers == {spk.target_count}."
            ),
        ))
    else:
        if spk.min_count is not None:
            out.append(_file_level_pbv(
                key="num_speakers",
                operator="ge",
                value=int(spk.min_count),
                reason=(
                    f"speakers.min_count={spk.min_count} → keep "
                    f"files where num_speakers >= {spk.min_count}."
                ),
            ))
        if spk.max_count is not None:
            out.append(_file_level_pbv(
                key="num_speakers",
                operator="le",
                value=int(spk.max_count),
                reason=(
                    f"speakers.max_count={spk.max_count} → keep "
                    f"files where num_speakers <= {spk.max_count}."
                ),
            ))
    return out


def _speaker_stages(intent: IntentCategories) -> list[StageRef]:
    """Emit segmenter-position speaker stages: SpeakerSeparation for
    SPLIT, and the degraded Sortformer-only fallback when SPLIT is
    incompatible with ``output_unit=='original_files'``.

    File-level diarization for ANNOTATE / FILTER is emitted earlier by
    :func:`_file_level_speaker_stages` so the num_speakers count is
    computed on whole files, not per VAD segment.

    No special wiring is needed for a downstream post-segmenter VAD trim:
    ``SpeakerSeparationStage`` returns a per-speaker waveform that is the
    full source duration with silence outside the speaker's regions (see
    ``speaker_separation_module/speaker_sep.py`` and the reference
    layout in ``advanced_pipelines/audio_data_filter``). Coordinates are
    therefore source-aligned end to end, so ``VAD(nested=False)`` can run
    immediately after this stage and its ``start_ms`` / ``end_ms`` will
    be valid against the same file that ``SegmentExtractionStage`` later
    opens.
    """

    spk = intent.speakers
    out: list[StageRef] = []

    if spk.mode == FilterMode.OFF:
        return out

    if spk.mode == FilterMode.SPLIT:
        # ``speakers.mode == SPLIT`` is a fan-out signal (one output task
        # per detected speaker). It is therefore incompatible with
        # ``output_unit == 'original_files'``, which promises one output
        # row per input file. The phase-sort would also place a cleaning-flow
        # ``SegmentConcatenationStage`` after ``SpeakerSeparationStage``,
        # where its ``nested_segments`` precondition can no longer be met.
        # Degrade to diarization-without-fan-out so we still surface speaker
        # metadata without breaking the file-level cardinality contract.
        if intent.segmentation.output_unit == "original_files":
            out.append(StageRef(
                stage="InferenceSortformerStage",
                params={},
                auto_inserted=True,
                insert_reason=(
                    "speakers.mode='split' is incompatible with "
                    "output_unit='original_files' (fan-out vs one-file-out); "
                    "degraded to diarization-only (Sortformer)."
                ),
            ))
            return out
        out.append(StageRef(
            stage="SpeakerSeparationStage",
            params={"exclude_overlaps": bool(spk.exclude_overlaps)},
        ))
        return out

    # ANNOTATE / FILTER are emitted upstream by
    # :func:`_file_level_speaker_stages` so num_speakers is computed on
    # the whole source file, not a per-VAD-segment row that would
    # almost always report a single speaker.
    return out


# ----------------------------------------------------------------------------
# Text policy (transcripts, alignment, WER)
# ----------------------------------------------------------------------------


def _text_stages(intent: IntentCategories) -> list[StageRef]:
    text = intent.text
    out: list[StageRef] = []

    # ``ALMDataBuilderStage`` requires ``task.data['segments']`` AND
    # ``task.data['words']`` to produce non-empty windows (see its card
    # ``inputs.data`` list). Only ``NeMoASRAlignerStage`` produces both —
    # ``InferenceAsrNemoStage`` returns transcripts without alignment.
    # For ``output_unit=='long_windows'`` we therefore force the aligner
    # regardless of ``text.transcript_source`` / ``text.word_timing`` so
    # the compiled pipeline doesn't silently produce empty manifests.
    if intent.segmentation.output_unit == "long_windows":
        forced_asr = StageRef(
            stage="NeMoASRAlignerStage",
            params={},
            auto_inserted=True,
            insert_reason=(
                "output_unit='long_windows' requires ASR alignment for "
                "ALMDataBuilderStage (needs both 'segments' and 'words'); "
                "forced NeMoASRAlignerStage."
            ),
        )
        out.append(forced_asr)
        # Skip the user-requested ASR path entirely so we don't end up
        # with two ASR stages in the IR. WER post-filtering still applies.
        if text.wer_mode != FilterMode.OFF:
            out.append(StageRef(stage="GetPairwiseWerStage", params={}))
            if text.wer_mode == FilterMode.FILTER and text.wer_max is not None:
                out.append(_preserve_by_value(
                    key="pairwise_wer",
                    operator="le",
                    value=float(text.wer_max),
                    reason=(
                        f"text.wer_max={text.wer_max} → keep rows where "
                        f"pairwise_wer <= {text.wer_max}."
                    ),
                ))
        return out

    if text.transcript_source == "generate":
        if text.word_timing:
            # NeMoASRAlignerStage dataclass already defaults model_name to
            # a real Parakeet checkpoint, so an empty params dict is safe.
            out.append(StageRef(
                stage="NeMoASRAlignerStage",
                params={},
            ))
        else:
            # WhisperX ASR is exposed in the clarifier (text.asr_backend ==
            # "whisper") but the actual ``WhisperXAsrStage`` is not yet in
            # the registry — emitting it would fail validation with
            # ``stage_not_registered``. Until that stage lands we
            # gracefully degrade to the NeMo Parakeet path; the
            # ``insert_reason`` makes the substitution visible in the IR
            # so the user (and the UI) can see what happened.
            #
            # NB: we intentionally do NOT emit WhisperXVADStage in the
            # fallback. It stays 1:1 and writes ``vad_segments`` as a list
            # that nothing downstream consumes here; if a Silero
            # ``VADSegmentationStage`` is already in the pipeline (e.g.
            # ``output_unit=='speech_segments'``) adding it would double
            # up the VAD work for no benefit.
            #
            # InferenceAsrNemoStage's dataclass defaults model_name to ""
            # and raises if not supplied. Seed a known-good Parakeet
            # checkpoint here so the IR runs out of the box; the form
            # owner can still override via answers (or future tuner).
            asr_ref = StageRef(
                stage="InferenceAsrNemoStage",
                params={"model_name": "nvidia/parakeet-tdt_ctc-1.1b"},
            )
            if text.asr_backend == "whisper":
                asr_ref = asr_ref.model_copy(update={
                    "auto_inserted": True,
                    "insert_reason": (
                        "asr_backend='whisper' requested but WhisperXAsrStage "
                        "is not yet registered; falling back to InferenceAsrNemoStage."
                    ),
                })
            out.append(asr_ref)

    if text.wer_mode != FilterMode.OFF:
        out.append(StageRef(stage="GetPairwiseWerStage", params={}))
        if text.wer_mode == FilterMode.FILTER and text.wer_max is not None:
            out.append(_preserve_by_value(
                key="pairwise_wer",
                operator="le",
                value=float(text.wer_max),
                reason=(
                    f"text.wer_max={text.wer_max} → keep rows where "
                    f"pairwise_wer <= {text.wer_max}."
                ),
            ))

    return out


# ----------------------------------------------------------------------------
# ALM packaging
# ----------------------------------------------------------------------------


def _alm_stages(intent: IntentCategories) -> list[StageRef]:
    if intent.segmentation.output_unit != "long_windows":
        return []
    window = intent.segmentation.long_window_sec or 120.0
    return [
        StageRef(
            stage="ALMDataBuilderStage",
            params={"target_window_duration": float(window)},
        ),
    ]


# ----------------------------------------------------------------------------
# Duration gates for the whole-file flow
# ----------------------------------------------------------------------------


def _whole_file_duration_filter_stages(intent: IntentCategories) -> list[StageRef]:
    """Honor ``duration_min_sec`` / ``duration_max_sec`` when no VAD will run.

    VAD already enforces per-segment ``min_duration_sec`` / ``max_duration_sec``
    for ``speech_segments`` / cleaning-flow / single-speaker / long-windows
    pipelines (see :func:`_vad_stage` and :func:`_post_segmenter_vad_trim`).
    But when ``output_unit == 'original_files'`` *without* a cleaning flow,
    no VAD is in the pipeline — the user's duration constraints would have
    been dropped on the floor pre-fix.

    To close that gap we emit two ``PreserveByValueStage`` rows against
    the file-level ``duration`` key (populated by ``MonoConversionStage``
    / ``ResampleAudioStage``):

    * ``duration_min_sec`` → operator ``ge``
    * ``duration_max_sec`` → operator ``le``

    Both can be present at the same time and produce a closed range.
    """

    seg = intent.segmentation
    if seg.output_unit != "original_files":
        return []
    if _is_cleaning_flow(intent):
        # The cleaning flow already pushes durations into nested VAD.
        return []
    out: list[StageRef] = []
    if seg.duration_min_sec is not None:
        out.append(_preserve_by_value(
            key="duration",
            operator="ge",
            value=float(seg.duration_min_sec),
            reason=(
                f"segmentation.duration_min_sec={seg.duration_min_sec} on "
                f"output_unit='original_files' (no VAD) → keep files where "
                f"duration >= {seg.duration_min_sec}."
            ),
        ))
    if seg.duration_max_sec is not None:
        out.append(_preserve_by_value(
            key="duration",
            operator="le",
            value=float(seg.duration_max_sec),
            reason=(
                f"segmentation.duration_max_sec={seg.duration_max_sec} on "
                f"output_unit='original_files' (no VAD) → keep files where "
                f"duration <= {seg.duration_max_sec}."
            ),
        ))
    return out


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def _dedupe_by_identity(stages: list[StageRef]) -> list[StageRef]:
    """Drop trailing/identical duplicates while keeping first occurrence."""

    seen: dict[str, dict[str, Any]] = {}
    out: list[StageRef] = []
    for ref in stages:
        key = ref.stage
        if key in seen and seen[key] == ref.params:
            continue
        seen[key] = dict(ref.params)
        out.append(ref)
    return out


def _coerce_params(
    stage_name: str,
    params: dict[str, Any],
    registry: CapabilityRegistry,
) -> dict[str, Any]:
    """Clamp numeric params to card min/max and drop unknown keys.

    Mirrors the legacy coercion in ``deterministic_planner._coerce_params``;
    centralized here so the selector and the planner agree.
    """

    entry = registry.get(stage_name)
    if entry is None:
        return params
    card: StageCard = entry.card
    specs = {p.name: p for p in card.params}
    cleaned: dict[str, Any] = {}
    for k, v in params.items():
        if v is None or k not in specs:
            continue
        spec = specs[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if spec.min is not None:
                v = max(spec.min, v)
            if spec.max is not None:
                v = min(spec.max, v)
        if spec.choices is not None and v not in spec.choices:
            continue
        if spec.type == "int" and isinstance(v, (int, float)) and not isinstance(v, bool):
            v = int(v)
        elif spec.type == "float" and isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        cleaned[k] = v
    return cleaned


__all__ = ["select_stages"]
