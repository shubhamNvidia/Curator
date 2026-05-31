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
"""IntentCategories — the structured shape every downstream module reads.

This file is the source of truth for the *ingredient* schema described in
``INTENT_V2.md``. The LLM and the clarifier together populate this object;
every other module (compiler, validator, dry-run) reads it as data and
never re-parses the original prompt.

The schema is intentionally **namespaced** (``intent.output.sample_rate``
instead of ``intent.sample_rate``) so each subsystem touches the smallest
slice it actually cares about. The single top-level fields are
``raw_prompt`` (preserved for audit / replay) and ``notes`` (free-form
provenance the clarifier and selector can append to).

There is **no** ``goal`` / ``task_profile`` field. The user picks
ingredients; defaults come from the prompt + dataset profile + sensible
per-question fallbacks. See ``INTENT_V2.md`` for the full design.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_curator.agentic.cards import CapabilityTag

# Re-exported aliases (kept for readability in submodels below).
SampleRateValue = int | Literal["any"] | None
ChannelsValue = Literal["mono", "stereo", "any"] | None
AudioFileFormat = Literal["wav", "flac", "ogg"]


class FilterMode(str, Enum):
    """How an analytical stage is allowed to behave.

    ``OFF``       — stage is not added.
    ``ANNOTATE``  — stage runs; its outputs are written to the manifest;
                    no row is ever dropped (numeric thresholds collapse to 0).
    ``FILTER``    — stage runs; rows that fail the threshold are dropped.
    ``SPLIT``     — stage runs and fans out (1:N), producing one row per
                    segment / speaker / window. The downstream row schema
                    *changes*.
    """

    OFF = "off"
    ANNOTATE = "annotate"
    FILTER = "filter"
    SPLIT = "split"


# ----------------------------------------------------------------------------
# Output format ingredients
# ----------------------------------------------------------------------------


class OutputFormat(BaseModel):
    """How each output file should look.

    ``sample_rate`` can be a concrete integer Hz, the literal ``"any"`` (the
    user said "I don't care" — never insert a resample), or ``None`` (unset).
    ``resample_input`` only matters when ``sample_rate`` is a concrete number:
    ``True`` means "convert every input file to that rate", ``False`` means
    "only accept inputs that already match — drop the rest".
    """

    model_config = ConfigDict(extra="ignore")

    sample_rate: SampleRateValue = Field(
        default=None,
        description="Target output sample rate (Hz). 'any' = pass-through.",
    )
    channels: ChannelsValue = Field(
        default=None,
        description="Channel layout: 'mono', 'stereo', or 'any'.",
    )
    audio_format: AudioFileFormat | None = Field(
        default="wav",
        description="Output container format. ``None`` means manifest only.",
    )
    resample_input: bool | None = Field(
        default=None,
        description=(
            "Gated follow-up to ``sample_rate``. When True, all inputs are "
            "resampled to the target. When False, inputs that already match "
            "the target are kept and others are dropped. ``None`` until the "
            "user picks."
        ),
    )


# ----------------------------------------------------------------------------
# Segmentation / speech ingredients
# ----------------------------------------------------------------------------


SegmentationUnit = Literal[
    "original_files",
    "speech_segments",
    "long_windows",
    "single_speaker_clips",
]


class Segmentation(BaseModel):
    """What a single output row represents and how speech is handled."""

    model_config = ConfigDict(extra="ignore")

    output_unit: SegmentationUnit = Field(
        default="original_files",
        description="What does one manifest row represent in the final output?",
    )
    duration_min_sec: float | None = Field(default=None, ge=0.0, le=3600.0)
    duration_max_sec: float | None = Field(default=None, ge=0.0, le=3600.0)
    speech_policy: FilterMode = Field(
        default=FilterMode.OFF,
        description=(
            "Only meaningful when ``output_unit == 'original_files'``. "
            "OFF = no VAD; ANNOTATE = run VAD and surface segments on each "
            "row; FILTER = drop rows with no detected speech (currently "
            "implemented as ANNOTATE + a documented TODO until a "
            "list-length filter is added)."
        ),
    )
    long_window_sec: float | None = Field(
        default=None,
        ge=1.0,
        le=600.0,
        description="ALM window length; required when output_unit == 'long_windows'.",
    )
    vad_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    speech_pad_ms: int | None = Field(default=None, ge=0, le=2000)


# ----------------------------------------------------------------------------
# Quality ingredients (filter ↔ annotate duality)
# ----------------------------------------------------------------------------


SigmosAxis = Literal["ovrl", "noise", "sig", "col", "disc", "loud", "reverb"]

# Keys a quality gate may target. UTMOS contributes one (``utmos_mos``);
# SIGMOS contributes seven (``sigmos_<axis>``); BandFilter contributes one
# string-valued key (``band_prediction``). Adding a new key here also
# requires the selector to recognize it so the upstream scoring stage gets
# pinned (see ``_quality_stages`` in ``stage_selector.py``).
QualityGateKey = Literal[
    "utmos_mos",
    "sigmos_ovrl",
    "sigmos_noise",
    "sigmos_sig",
    "sigmos_col",
    "sigmos_disc",
    "sigmos_loud",
    "sigmos_reverb",
    "band_prediction",
]

QualityGateOperator = Literal["lt", "le", "eq", "ne", "ge", "gt"]


class QualityGate(BaseModel):
    """A single ``PreserveByValueStage`` rule against a quality-score key.

    The selector now treats every UTMOS / SIGMOS stage as a *pure
    annotator* (pins ``mos_threshold = 0`` / all SIGMOS axis thresholds to
    ``0.0`` so nothing is dropped at score time) and routes every drop
    decision through ``PreserveByValueStage``. That gives the user the
    full operator matrix on every metric — "keep MOS ≥ 4" is the common
    case but "drop pristine clips" ("``utmos_mos lt 4``") or "narrow-band
    only" ("``band_prediction eq narrow_band``") become first-class.

    Two ways to populate gates:

    * Implicit / legacy — keep using ``mos_threshold`` /
      ``sigmos_thresholds`` / ``band_value``; the selector compiles each
      into the equivalent ``ge`` / ``eq`` gate automatically.
    * Explicit — push a :class:`QualityGate` into ``Quality.gates``. This
      is the only way to express ``lt`` / ``gt`` / ``ne`` and is what
      the extractor LLM should reach for when the user says "less than",
      "more than", "drop high quality", etc.
    """

    model_config = ConfigDict(extra="ignore")

    key: QualityGateKey = Field(
        ...,
        description=(
            "Which task.data field to gate on. Must match a key produced "
            "by an upstream scoring stage."
        ),
    )
    operator: QualityGateOperator = Field(
        ...,
        description=(
            "Comparison applied as ``task.data[key] <operator> value``. "
            "The task is KEPT when the comparison is True; dropped "
            "otherwise."
        ),
    )
    value: float | str = Field(
        ...,
        description=(
            "Threshold for MOS-style keys (0.0-5.0 ACR scale) or the "
            "literal class label for ``band_prediction`` "
            "(``'narrow_band'`` / ``'full_band'``)."
        ),
    )


class Quality(BaseModel):
    """Naturalness/MOS/bandwidth filters."""

    model_config = ConfigDict(extra="ignore")

    mos: FilterMode = Field(
        default=FilterMode.OFF,
        description="UTMOS naturalness gate.",
    )
    mos_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description=(
            "UTMOS floor when ``mos == FILTER``. Compiled into "
            "``QualityGate(utmos_mos, ge, mos_threshold)``."
        ),
    )
    sigmos: FilterMode = Field(
        default=FilterMode.OFF,
        description="Multi-axis SIGMOS gate.",
    )
    sigmos_axes: list[SigmosAxis] = Field(
        default_factory=list,
        description=(
            "Axes the user cares about. When ``sigmos == ANNOTATE`` every "
            "listed axis is scored with threshold 0; when ``FILTER``, "
            "each axis is compiled into a ``QualityGate(sigmos_<axis>, "
            "ge, sigmos_thresholds[<axis>])`` rule."
        ),
    )
    sigmos_thresholds: dict[SigmosAxis, float] = Field(
        default_factory=dict,
        description=(
            "Per-axis floor when ``sigmos == FILTER``. Each entry is "
            "compiled into a ``ge`` gate on the matching ``sigmos_<axis>`` "
            "key. Use ``gates`` directly for non-``ge`` operators."
        ),
    )
    band: FilterMode = Field(
        default=FilterMode.OFF,
        description=(
            "Bandwidth classifier filter. Only OFF or FILTER today — "
            "BandFilterStage has no annotate-only mode in the catalog. "
            "When FILTER, the selector compiles it into a ``QualityGate("
            "band_prediction, eq, band_value)`` rule."
        ),
    )
    band_value: Literal["narrow_band", "full_band"] | None = Field(
        default=None,
        description="Which bandwidth class to keep when band == FILTER.",
    )
    band_min_hz: float | None = Field(
        default=None,
        ge=0.0,
        le=24_000.0,
        description="Informational only — kept for prompt-language inference.",
    )
    gates: list[QualityGate] = Field(
        default_factory=list,
        description=(
            "Free-form quality gates. Use this to express any operator "
            "OTHER than ``ge`` on a MOS axis — e.g. ``QualityGate("
            "key='utmos_mos', operator='lt', value=2.0)`` for an "
            "adversarial / noisy-only corpus. The selector appends one "
            "``PreserveByValueStage`` per gate AFTER the legacy fields, "
            "so user-supplied gates always run last."
        ),
    )


# ----------------------------------------------------------------------------
# Speakers ingredients
# ----------------------------------------------------------------------------


class Speakers(BaseModel):
    """How to handle multi-speaker audio."""

    model_config = ConfigDict(extra="ignore")

    mode: FilterMode = Field(
        default=FilterMode.OFF,
        description=(
            "OFF = nothing. ANNOTATE = diarize and tag rows. "
            "FILTER = diarize and drop rows whose speaker count is wrong. "
            "SPLIT = fan out to one row per speaker (SpeakerSeparation)."
        ),
    )
    target_count: int | None = Field(
        default=None,
        ge=1,
        le=64,
        description=(
            "Required when mode == FILTER and exact-N is desired. "
            "Maps to PreserveByValueStage(num_speakers, operator='eq')."
        ),
    )
    min_count: int | None = Field(
        default=None,
        ge=1,
        le=64,
        description=(
            "Lower bound on the speaker count when mode == FILTER. "
            "Maps to PreserveByValueStage(num_speakers, operator='ge'). "
            "Combine with ``max_count`` to express a closed range, or "
            "with ``target_count`` for exclusion below an exact match."
        ),
    )
    max_count: int | None = Field(
        default=None,
        ge=1,
        le=64,
        description=(
            "Upper bound on the speaker count when mode == FILTER. "
            "Maps to PreserveByValueStage(num_speakers, operator='le')."
        ),
    )
    exclude_overlaps: bool = Field(
        default=True,
        description="Passed through to SpeakerSeparationStage when mode == SPLIT.",
    )


# ----------------------------------------------------------------------------
# Text / transcription ingredients
# ----------------------------------------------------------------------------


class TextPolicy(BaseModel):
    """Transcript generation, word timing, WER policing."""

    model_config = ConfigDict(extra="ignore")

    transcript_source: Literal["off", "generate", "existing"] = Field(
        default="off",
        description="Where transcripts come from — none, ASR, or an existing manifest column.",
    )
    asr_backend: Literal["nemo", "whisper", "auto"] = Field(
        default="auto",
        description="Only relevant when transcript_source == 'generate'.",
    )
    word_timing: bool = Field(
        default=False,
        description="Add word-level timestamps (switches generate → alignment stage).",
    )
    wer_mode: FilterMode = Field(
        default=FilterMode.OFF,
        description="WER policy vs. an existing transcript — annotate or filter.",
    )
    wer_max: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="WER ceiling when wer_mode == FILTER.",
    )


# ----------------------------------------------------------------------------
# Cross-cutting policy / budget
# ----------------------------------------------------------------------------


class Policy(BaseModel):
    """Cross-cutting policies that gate the validator and the runner."""

    model_config = ConfigDict(extra="ignore")

    commercial_only: bool = Field(
        default=False,
        description="If true, the license gate blocks NC/GPL/Unknown models.",
    )
    privacy_mode: bool = Field(
        default=False,
        description="If true, the runner refuses remote downloads / telemetry (design only).",
    )
    budget_gpu_hours: float | None = Field(default=None, ge=0.0, le=10_000.0)
    budget_disk_gb: float | None = Field(default=None, ge=0.0, le=100_000.0)
    target_directory: str | None = Field(
        default=None,
        description="Where to write outputs. Overridden by SinkSpec when set.",
    )
    target_manifest_filename: str = Field(
        default="manifest.jsonl",
        description="Final manifest filename written into target_directory.",
    )


# ----------------------------------------------------------------------------
# Top-level IntentCategories
# ----------------------------------------------------------------------------


class IntentCategories(BaseModel):
    """Structured ingredient list extracted from a natural-language prompt.

    Every downstream module operates on this object — the original prompt
    is preserved in ``raw_prompt`` for audit but never re-parsed.

    The submodels follow the question DAG documented in
    ``INTENT_V2.md``. Each submodel can be mutated independently of the
    others, which is the property the single-shot ingredient form relies
    on.
    """

    model_config = ConfigDict(extra="ignore")

    output: OutputFormat = Field(default_factory=OutputFormat)
    segmentation: Segmentation = Field(default_factory=Segmentation)
    quality: Quality = Field(default_factory=Quality)
    speakers: Speakers = Field(default_factory=Speakers)
    text: TextPolicy = Field(default_factory=TextPolicy)
    policy: Policy = Field(default_factory=Policy)

    raw_prompt: str | None = Field(
        default=None,
        description="Original user prompt — preserved for replay/audit.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form provenance: heuristic source tags, unsupported-feature "
            "warnings, clarifier answers. Always additive."
        ),
    )

    @model_validator(mode="after")
    def _consistency(self) -> IntentCategories:
        seg = self.segmentation
        if (
            seg.duration_min_sec is not None
            and seg.duration_max_sec is not None
            and seg.duration_min_sec > seg.duration_max_sec
        ):
            msg = (
                f"segmentation.duration_min_sec ({seg.duration_min_sec}) > "
                f"duration_max_sec ({seg.duration_max_sec})"
            )
            raise ValueError(msg)
        if seg.output_unit == "long_windows" and seg.long_window_sec is None:
            # Don't crash — just record a note so the selector can default.
            self.notes.append(
                "segmentation.output_unit=long_windows without long_window_sec; "
                "selector will default to 120s."
            )
        spk = self.speakers
        if (
            spk.mode == FilterMode.FILTER
            and spk.target_count is None
            and spk.max_count is None
            and spk.min_count is None
        ):
            self.notes.append(
                "speakers.mode=FILTER without target_count / min_count / "
                "max_count; selector will degrade to ANNOTATE so nothing is "
                "dropped silently."
            )
        if (
            spk.min_count is not None
            and spk.max_count is not None
            and spk.min_count > spk.max_count
        ):
            msg = (
                f"speakers.min_count ({spk.min_count}) > max_count "
                f"({spk.max_count})"
            )
            raise ValueError(msg)

        # ---- speech_segments + speakers=SPLIT  →  single_speaker_clips ----
        # When the user asks for speaker fan-out AND a per-segment output
        # unit, the canonical pipeline is the one the selector already
        # emits for ``single_speaker_clips``:
        #
        #   SpeakerSeparation → VAD(nested=False, min/max) → quality → ASR
        #
        # Leaving the intent as ``speech_segments`` forces the validator
        # to auto-insert a SegmentConcatenationStage *before*
        # SpeakerSeparation, which throws away the duration window and
        # makes per-speaker scoring useless (see regression run
        # 06578e4b178f11e3). Coerce now so both routes converge on the
        # same compiled pipeline.
        if (
            seg.output_unit == "speech_segments"
            and spk.mode == FilterMode.SPLIT
        ):
            self.segmentation = seg.model_copy(update={"output_unit": "single_speaker_clips"})
            self.notes.append(
                "segmentation.output_unit coerced from 'speech_segments' to "
                "'single_speaker_clips' because speakers.mode=SPLIT means "
                "per-speaker fan-out (SpeakerSeparation must run before VAD "
                "and quality scoring, not after a Concat)."
            )

        # ---- single_speaker_clips ↔ speakers=SPLIT (inverse) ----------------
        # The opposite asymmetry: ``output_unit=single_speaker_clips``
        # only makes sense when SpeakerSeparation is going to fan rows
        # out. If the selector won't emit that stage (because
        # ``speakers.mode`` is OFF / FILTER / ANNOTATE) the SanityCritic
        # later raises ``single_speaker_clips_missing_separation`` and
        # the build dies. This happened in run c27f2786eaad1d75 where
        # the Plan Critic patched the output_unit but the user had
        # locked ``speakers.mode=filter``, leaving the intent in a
        # state no selector path could satisfy.
        #
        # Resolution depends on whether the user has shown intent for
        # any file-level speaker handling:
        #
        # - ``mode=OFF``  → quietly upgrade to ``SPLIT``; the
        #   per-clip output unit was the explicit signal.
        # - ``mode=FILTER`` or ``mode=ANNOTATE`` → these are file-level
        #   diarization knobs the user usually picks deliberately; revert
        #   ``output_unit`` to ``speech_segments`` instead. The audit
        #   note explains the revert so the critic UI can surface it.
        if (
            self.segmentation.output_unit == "single_speaker_clips"
            and spk.mode != FilterMode.SPLIT
        ):
            if spk.mode == FilterMode.OFF:
                self.speakers = spk.model_copy(update={"mode": FilterMode.SPLIT})
                self.notes.append(
                    "speakers.mode coerced from OFF to SPLIT because "
                    "segmentation.output_unit=single_speaker_clips needs "
                    "per-speaker fan-out (SpeakerSeparationStage)."
                )
            else:
                # FILTER / ANNOTATE — file-level intent wins.
                self.segmentation = self.segmentation.model_copy(
                    update={"output_unit": "speech_segments"}
                )
                self.notes.append(
                    "segmentation.output_unit reverted to 'speech_segments' "
                    f"because speakers.mode={spk.mode.value} is incompatible "
                    "with 'single_speaker_clips' (only speakers.mode=SPLIT "
                    "produces per-speaker fan-out). To get per-speaker clips, "
                    "set speakers.mode=SPLIT."
                )
        return self


# ----------------------------------------------------------------------------
# Capability requirements & gap reporting
#
# These are intentionally kept at the public surface so the multi-agent
# team planner and NAT extractor can keep iterating without a coupled
# rewrite. The implementation now reads from the V2 submodels but emits
# the same CapabilityRequirement shape as before.
# ----------------------------------------------------------------------------


class CapabilityRequirement(BaseModel):
    """One row of the deterministic intent → capabilities mapping."""

    model_config = ConfigDict(extra="forbid")

    intent_field: str
    capability: CapabilityTag
    reason: str
    required: bool = True


class IntentGap(BaseModel):
    """A capability the user asked for that the catalog cannot satisfy."""

    model_config = ConfigDict(extra="forbid")

    intent_field: str
    requested_value: Any
    missing_capability: CapabilityTag
    suggestion: str


def required_capabilities(intent: IntentCategories) -> list[CapabilityRequirement]:
    """Derive the capability tags an intent implies.

    This is the legacy capability-list surface used by the multi-agent
    team planner. The deterministic compiler uses ``stage_selector``
    directly, which has a strictly richer view — but both must agree on
    what is required, so this function is intentionally aligned with
    the §4 matrix in ``INTENT_V2.md``.
    """

    reqs: list[CapabilityRequirement] = []

    # ---- Segmentation / speech --------------------------------------------
    unit = intent.segmentation.output_unit
    if unit == "speech_segments":
        reqs.append(CapabilityRequirement(
            intent_field="segmentation.output_unit",
            capability=CapabilityTag.VAD,
            reason="Need VAD-style segmentation for per-segment output.",
        ))
        reqs.append(CapabilityRequirement(
            intent_field="segmentation.output_unit",
            capability=CapabilityTag.SEGMENT_EXTRACT,
            reason="Need to write per-segment audio files.",
        ))
    elif unit == "single_speaker_clips":
        # SpeakerSeparationStage is the segmenter here — it consumes
        # whole-file audio and fans out one task per speaker. The
        # cleaning flow (VAD nested → filters → concat) is reserved for
        # ``output_unit == 'original_files'``; we don't run an upfront
        # VAD here because it would double-segment the audio.
        reqs.append(CapabilityRequirement(
            intent_field="segmentation.output_unit",
            capability=CapabilityTag.SPEAKER_SEPARATION,
            reason="Single-speaker clips require speaker separation.",
        ))
        reqs.append(CapabilityRequirement(
            intent_field="segmentation.output_unit",
            capability=CapabilityTag.SEGMENT_EXTRACT,
            reason="Need to write per-clip audio files.",
        ))
        # VAD comes back as a *post-segmenter trim* whenever the user
        # gave us a per-clip duration cap or a speech-presence policy:
        # the selector emits ``VAD(nested=False)`` after
        # ``SpeakerSeparationStage`` to enforce
        # ``duration_min_sec`` / ``duration_max_sec`` /
        # ``speech_policy`` on each fan-out clip. Capabilities must
        # reflect that so the multi-agent picker resolves VAD upfront.
        if (
            intent.segmentation.duration_min_sec is not None
            or intent.segmentation.duration_max_sec is not None
            or intent.segmentation.speech_policy != FilterMode.OFF
        ):
            reqs.append(CapabilityRequirement(
                intent_field="segmentation.output_unit",
                capability=CapabilityTag.VAD,
                reason=(
                    "Post-segmenter VAD trim enforces duration / "
                    "speech-policy constraints on each single-speaker clip."
                ),
            ))
    elif unit == "long_windows":
        reqs.append(CapabilityRequirement(
            intent_field="segmentation.output_unit",
            capability=CapabilityTag.ALM_PACKAGE,
            reason="Long-audio window packaging.",
        ))
        if (
            intent.segmentation.duration_min_sec is not None
            or intent.segmentation.duration_max_sec is not None
            or intent.segmentation.speech_policy != FilterMode.OFF
        ):
            reqs.append(CapabilityRequirement(
                intent_field="segmentation.output_unit",
                capability=CapabilityTag.VAD,
                reason=(
                    "Post-segmenter VAD trim enforces duration / "
                    "speech-policy constraints on each long-window clip."
                ),
            ))
    elif intent.segmentation.speech_policy != FilterMode.OFF:
        reqs.append(CapabilityRequirement(
            intent_field="segmentation.speech_policy",
            capability=CapabilityTag.VAD,
            reason="Speech-presence policy requires VAD.",
        ))

    # ---- Quality ----------------------------------------------------------
    if intent.quality.mos != FilterMode.OFF or intent.quality.sigmos != FilterMode.OFF:
        reqs.append(CapabilityRequirement(
            intent_field="quality.mos",
            capability=CapabilityTag.QUALITY_FILTER_MOS,
            reason="User wants a MOS-family gate (annotate or filter).",
        ))
    if intent.quality.band == FilterMode.FILTER and intent.quality.band_value is not None:
        reqs.append(CapabilityRequirement(
            intent_field="quality.band",
            capability=CapabilityTag.QUALITY_FILTER_BAND,
            reason="User wants a bandwidth gate.",
        ))

    # ---- Speakers ---------------------------------------------------------
    spk = intent.speakers
    if spk.mode == FilterMode.SPLIT:
        reqs.append(CapabilityRequirement(
            intent_field="speakers.mode",
            capability=CapabilityTag.SPEAKER_SEPARATION,
            reason="Speaker SPLIT: fan out to one row per speaker.",
        ))
    elif spk.mode in {FilterMode.ANNOTATE, FilterMode.FILTER}:
        reqs.append(CapabilityRequirement(
            intent_field="speakers.mode",
            capability=CapabilityTag.SPEAKER_DIARIZATION,
            reason="Speaker mode requires diarization.",
        ))

    # ---- Text -------------------------------------------------------------
    text = intent.text
    if text.transcript_source == "generate":
        reqs.append(CapabilityRequirement(
            intent_field="text.transcript_source",
            capability=CapabilityTag.ASR,
            reason="User wants ASR-generated transcripts.",
        ))
        if text.word_timing:
            reqs.append(CapabilityRequirement(
                intent_field="text.word_timing",
                capability=CapabilityTag.ASR_ALIGN,
                reason="User wants word-level alignment.",
            ))
    if text.wer_mode != FilterMode.OFF:
        reqs.append(CapabilityRequirement(
            intent_field="text.wer_mode",
            capability=CapabilityTag.WER,
            reason="WER policy (annotate or filter) requires a WER metric.",
        ))

    # ---- ALM packaging via long_windows ----------------------------------
    if intent.segmentation.output_unit == "long_windows":
        # ALM packaging benefits from text + alignment; reqs already
        # added above will pull those in when the user opts in.
        pass

    # Dedupe capabilities: the same tag can be implied by two ingredients
    # (e.g. ``output_unit=single_speaker_clips`` and ``speakers.mode=SPLIT``
    # both imply SPEAKER_SEPARATION). The multi-agent picker iterates per
    # row, so duplicates would burn an extra LLM call for the same capability.
    seen: set[CapabilityTag] = set()
    unique: list[CapabilityRequirement] = []
    for req in reqs:
        if req.capability in seen:
            continue
        seen.add(req.capability)
        unique.append(req)
    return unique


__all__ = [
    "AudioFileFormat",
    "CapabilityRequirement",
    "ChannelsValue",
    "FilterMode",
    "IntentCategories",
    "IntentGap",
    "OutputFormat",
    "Policy",
    "Quality",
    "SampleRateValue",
    "SegmentationUnit",
    "Segmentation",
    "SigmosAxis",
    "Speakers",
    "TextPolicy",
    "required_capabilities",
]
