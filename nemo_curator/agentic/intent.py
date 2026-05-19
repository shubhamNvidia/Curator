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
"""IntentCategories — the fixed JSON shape the LLM extracts from a prompt.

This is the *only* point where the LLM sees free-form user text. The
extraction tool's job is to populate this schema; every downstream step
(planner, validator, compiler, runner, critic) operates on the structured
intent, not the prompt.

The schema only includes fields that the current 28-stage catalog can
actually satisfy. When new stages land (LID, emotion, augmentation, MOS
variants, SNR, etc.) the schema will grow at the same time as the stages
it serves. The agent does not pretend to support intents the catalog
cannot honor; the NAT extraction tool reports a clean "unsupported
intent" finding when the prompt asks for capabilities the catalog
doesn't expose.

See ``ADV/adv_phase2/AUDIO_INTERNALS.md`` section "Predefined intent
categories schema" for the rationale.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_curator.agentic.cards import CapabilityTag

# Sentinel for "user didn't constrain this" vs. "user said any". ``None`` =
# unset; the literal ``"any"`` = explicitly any (consume but don't filter).
SampleRateValue = int | Literal["any"] | None
ChannelsValue = Literal["mono", "stereo", "any"] | None
OutputFormat = Literal["wav", "flac", "ogg"]


class IntentCategories(BaseModel):
    """Structured intent extracted from a natural-language prompt.

    Field semantics:

    - ``None``  → user did not mention this category. Planner ignores it.
    - sentinel ``"any"`` → user said "any" / "I don't care". Planner accepts
      anything but does not insert filters.
    - concrete value → planner must satisfy it; if no covering capability
      exists in the registry, the agent emits a deterministic finding and
      refuses to plan.

    Numeric fields use ``Optional[float]`` instead of an ``"any"`` sentinel
    because their natural unset value is ``None``.

    ``extra="ignore"`` is intentional: the agent often hallucinates plausible
    grouping fields (e.g. ``budgets: {gpu_hours, disk_gb}``) and we'd rather
    discard them than abort the planning loop. Strict validation still
    catches type errors on the fields we DO recognize.
    """

    model_config = ConfigDict(extra="ignore")

    # --- Format ----------------------------------------------------------
    sample_rate: SampleRateValue = Field(
        default=None,
        description="Target sample rate in Hz, e.g. 16000 or 48000. 'any' to accept any.",
    )
    channels: ChannelsValue = Field(
        default=None,
        description="Channel constraint: mono, stereo, or any.",
    )
    output_format: OutputFormat = Field(
        default="wav",
        description="Output audio format for extracted clips.",
    )

    # --- Duration / segmentation ----------------------------------------
    duration_min_sec: float | None = Field(default=None, ge=0.0, le=3600.0)
    duration_max_sec: float | None = Field(default=None, ge=0.0, le=3600.0)
    output_extract_clips: bool = Field(
        default=True,
        description=(
            "If true, fan out to one file per VAD/diarization segment. "
            "If false, keep the original file boundaries."
        ),
    )

    # --- Quality (MOS-family only — UTMOS/SIGMOS/Band today) ------------
    quality_mos_min: float | None = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Floor on UTMOS/SIGMOS-style mean opinion score.",
    )
    quality_bandwidth_min_hz: float | None = Field(
        default=None,
        ge=0.0,
        le=24_000.0,
        description="Floor on detected effective bandwidth, via BandFilterStage.",
    )

    # --- Speakers -------------------------------------------------------
    speakers: int | Literal["any"] | None = Field(
        default=None,
        description=(
            "Exact number of speakers required per output clip, or 'any'. "
            "When set to 1, the planner routes to SpeakerSeparationStage "
            "(safe default: passes through single-speaker input, fans out "
            "multi-speaker input, overlaps excluded by default). When >1, "
            "the planner pairs InferenceSortformerStage with a value-filter "
            "on num_speakers."
        ),
    )

    # --- ASR & alignment -----------------------------------------------
    need_asr: bool = False
    asr_wer_max: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="If set, require a reference transcript and drop clips whose ASR-WER exceeds this value.",
    )
    need_word_alignment: bool = False
    need_diarization: bool = False

    # --- Audio Language Model packaging --------------------------------
    alm_window_sec: float | None = Field(
        default=None,
        ge=1.0,
        le=600.0,
        description="If set, package output into ALM windows of this length.",
    )
    alm_overlap_dedupe: bool = False

    # --- Policy / budget -----------------------------------------------
    commercial_only: bool = Field(
        default=False,
        description="If true, license gate blocks NC/GPL/Unknown models.",
    )
    budget_gpu_hours: float | None = Field(default=None, ge=0.0, le=10_000.0)
    budget_disk_gb: float | None = Field(default=None, ge=0.0, le=100_000.0)
    privacy_mode: bool = Field(
        default=False,
        description="No telemetry, no remote model downloads (design only; not enforced yet).",
    )

    # --- Output layout --------------------------------------------------
    target_directory: str | None = None
    target_manifest_filename: str = "manifest.jsonl"

    # --- Free-form provenance ------------------------------------------
    raw_prompt: str | None = Field(
        default=None,
        description="The original user prompt; preserved for replay and audit.",
    )
    notes: list[str] = Field(default_factory=list)

    # ---- Validators ---------------------------------------------------

    @model_validator(mode="after")
    def _duration_order(self) -> IntentCategories:
        if (
            self.duration_min_sec is not None
            and self.duration_max_sec is not None
            and self.duration_min_sec > self.duration_max_sec
        ):
            msg = f"duration_min_sec ({self.duration_min_sec}) > duration_max_sec ({self.duration_max_sec})"
            raise ValueError(msg)
        return self

    @field_validator("sample_rate")
    @classmethod
    def _sample_rate_known(cls, v: SampleRateValue) -> SampleRateValue:
        if isinstance(v, int) and v not in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
            msg = f"sample_rate {v} is not a common rate; the planner may have trouble routing it"
            raise ValueError(msg)
        return v


# ----------------------------------------------------------------------------
# Capability requirements & gap reporting
# ----------------------------------------------------------------------------


class CapabilityRequirement(BaseModel):
    """One row of the deterministic categories → capabilities mapping.

    Used by the planner *and* by the gap reporter. If no stage in the
    registry exposes the required ``CapabilityTag`` (either as ``capabilities``
    or ``also_handles``), the agent emits a gap finding instead of
    hallucinating a stage.
    """

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
    """Deterministic mapping from non-default :class:`IntentCategories` fields to capabilities.

    The agent's planner uses this to assemble a candidate set; the registry
    then resolves each capability to a concrete stage (or reports a gap).
    """

    reqs: list[CapabilityRequirement] = []

    if (
        intent.duration_min_sec is not None
        or intent.duration_max_sec is not None
        or intent.output_extract_clips
    ):
        reqs.append(CapabilityRequirement(
            intent_field="output_extract_clips",
            capability=CapabilityTag.VAD,
            reason="Need VAD-style segmentation to honor duration limits / extract clips.",
        ))
        reqs.append(CapabilityRequirement(
            intent_field="output_extract_clips",
            capability=CapabilityTag.SEGMENT_EXTRACT,
            reason="Need to write per-segment audio files.",
        ))

    if intent.quality_mos_min is not None:
        reqs.append(CapabilityRequirement(
            intent_field="quality_mos_min",
            capability=CapabilityTag.QUALITY_FILTER_MOS,
            reason="User wants a MOS floor.",
        ))

    if intent.quality_bandwidth_min_hz is not None:
        reqs.append(CapabilityRequirement(
            intent_field="quality_bandwidth_min_hz",
            capability=CapabilityTag.QUALITY_FILTER_BAND,
            reason="User wants a minimum effective bandwidth.",
        ))

    if intent.speakers == 1:
        # Single-speaker output uses SpeakerSeparationStage as the safe
        # default — it passes through if the source is already 1 speaker
        # and fans out otherwise, with overlap regions excluded by default.
        reqs.append(CapabilityRequirement(
            intent_field="speakers",
            capability=CapabilityTag.SPEAKER_SEPARATION,
            reason="Single-speaker output: SpeakerSeparationStage handles both single- and multi-speaker source data deterministically.",
        ))
    elif isinstance(intent.speakers, int) and intent.speakers > 1:
        # N-speaker enforcement uses diarization + a value-filter on
        # num_speakers; no separation needed because we keep the original audio.
        reqs.append(CapabilityRequirement(
            intent_field="speakers",
            capability=CapabilityTag.SPEAKER_DIARIZATION,
            reason="N-speaker constraint enforced via diarization + num_speakers value-filter.",
        ))

    if intent.need_asr or intent.asr_wer_max is not None:
        reqs.append(CapabilityRequirement(
            intent_field="need_asr",
            capability=CapabilityTag.ASR,
            reason="User wants ASR transcripts.",
        ))
        if intent.asr_wer_max is not None:
            reqs.append(CapabilityRequirement(
                intent_field="asr_wer_max",
                capability=CapabilityTag.WER,
                reason="Need WER metric to enforce wer-max filter.",
            ))

    if intent.need_word_alignment:
        reqs.append(CapabilityRequirement(
            intent_field="need_word_alignment",
            capability=CapabilityTag.ASR_ALIGN,
            reason="User wants word-level alignment.",
        ))

    # Diarization is only required when the user explicitly asks for it.
    # Speaker-count routing already lives in the dedicated branch above.
    if intent.need_diarization:
        reqs.append(CapabilityRequirement(
            intent_field="need_diarization",
            capability=CapabilityTag.SPEAKER_DIARIZATION,
            reason="User wants per-speaker timestamps.",
        ))

    if intent.alm_window_sec:
        reqs.append(CapabilityRequirement(
            intent_field="alm_window_sec",
            capability=CapabilityTag.ALM_PACKAGE,
            reason="User wants ALM-formatted windows.",
        ))
        if intent.alm_overlap_dedupe:
            reqs.append(CapabilityRequirement(
                intent_field="alm_overlap_dedupe",
                capability=CapabilityTag.ALM_OVERLAP,
                reason="User wants overlap-based dedup of ALM windows.",
            ))

    return reqs


__all__ = [
    "CapabilityRequirement",
    "IntentCategories",
    "IntentGap",
    "required_capabilities",
]
