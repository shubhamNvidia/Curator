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
"""Adaptive, LLM-driven clarifier.

This module replaces the static ingredient-picker form. The flow is:

1. ``extract_intent`` (already done upstream) produces an
   :class:`IntentCategories` with whatever the LLM could pull from the
   prompt + dataset profile.
2. :func:`analyze_gaps` is a *deterministic* pass that decides which
   fields are still essential, missing, or worth confirming. It encodes
   the policy "only ask what the user can't already be assumed to want".
3. :func:`compose_smart_form` either asks the LLM to phrase those gaps
   into 1-5 user-friendly questions (with inline conditional follow-ups),
   or — if the LLM is unavailable / errors out — falls back to a
   deterministic template that produces equivalent questions.
4. The legacy ingredient form is still produced and embedded as
   ``advanced_form`` so power users can open the "Show all options"
   expander to override anything the smart flow didn't surface.

The smart form is rendered by ``web.py`` as:

   ┌─ Inferred (collapsed) ────────────────────────────────────┐
   │  N settings extracted from your prompt + dataset profile  │
   └───────────────────────────────────────────────────────────┘
   ┌─ Quick questions ─────────────────────────────────────────┐
   │  1. Output sample rate?                                   │
   │  2. Need duration constraints? [No] [Yes ▾ min / max]     │
   └───────────────────────────────────────────────────────────┘
   ▶ Show all options (advanced)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from nemo_curator.agentic.cards import DatasetCard
from nemo_curator.agentic.clarifier import (
    ClarificationForm,
    apply_answers,
    apply_profile_prefills,
    build_clarification_form,
    infer_intent_from_prompt,
)
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
)
from nemo_curator.agentic.llm import LLMClient, Message

# ----------------------------------------------------------------------------
# Public schemas (returned to the web UI)
# ----------------------------------------------------------------------------


class InferredChip(BaseModel):
    """One settled-by-LLM-or-profile fact, shown collapsed at the top."""

    model_config = ConfigDict(extra="forbid")

    intent_path: str
    label: str
    value: Any
    source: Literal["prompt", "profile", "default"]
    why: str


class SmartOption(BaseModel):
    """One pickable answer to a smart question."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    description: str | None = None
    value: Any = None
    apply: dict[str, Any] | None = None
    is_freeform: bool = False
    freeform_kind: Literal[
        "int_hz", "float_sec", "speaker_count", "duration_range", "free", "int", "float",
    ] | None = None
    freeform_placeholder: str | None = None
    reveals: list[str] = Field(
        default_factory=list,
        description="IDs of follow-up questions to reveal when this option is picked.",
    )


class SmartQuestion(BaseModel):
    """One adaptive question, possibly with conditional follow-ups."""

    model_config = ConfigDict(extra="forbid")

    id: str
    intent_path: str
    title: str
    detail: str | None = None
    options: list[SmartOption]
    follow_ups: list["SmartQuestion"] = Field(default_factory=list)
    visible_when: dict[str, Any] | None = Field(
        default=None,
        description=(
            "When non-null this question only renders if the parent answer "
            "matches. Shape: {'parent_id': 'q_duration_enable', "
            "'reveal_when_option': 'yes'}."
        ),
    )
    prefill: Any = None
    is_freeform_root: bool = False
    freeform_kind: Literal[
        "int_hz", "float_sec", "speaker_count", "duration_range", "free", "int", "float",
    ] | None = None
    freeform_placeholder: str | None = None


class SmartForm(BaseModel):
    """The whole payload returned by ``/api/plan``."""

    model_config = ConfigDict(extra="forbid")

    inferred: list[InferredChip] = Field(default_factory=list)
    questions: list[SmartQuestion] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    profile_summary: dict[str, Any] | None = None
    advanced_form: ClarificationForm
    intent: IntentCategories


# ----------------------------------------------------------------------------
# Gap analyzer — pure code, the policy "what is essential / worth asking"
# ----------------------------------------------------------------------------


@dataclass
class Gap:
    """One question worth asking the user."""

    intent_path: str
    category: Literal[
        "essential_missing",
        "essential_uncertain",
        "prompt_mentioned",
        "profile_disagreement",
    ]
    reason: str
    suggested_value: Any = None
    follow_ups: list["Gap"] = field(default_factory=list)


_DURATION_TERMS = (
    "duration", "between", "seconds", "minutes", "minimum", "maximum",
    "shorter", "longer", " min ", " max ", "at least", "at most",
    "clip", "short ", "long ", "segment", " sec", " min.",
)
_SPEAKER_TERMS = (
    "speaker", "diariz", "who said", "meeting", "interview", "multi-speaker",
    "single speaker", "multiple speaker", "one speaker", "two speaker",
)
_QUALITY_TERMS = (
    "noisy", "clean", "quality", "studio", "broadcast", "naturalness",
    "mos", "sigmos", "noise", "denoise", "filter low", "pristine",
)
_TRANSCRIPT_TERMS = (
    "transcript", "transcribe", "stt", "asr", "subtitles", "captions",
    "alignment", "force-align", "word timing", "word-level",
)
_PHONE_TERMS = ("phone", "telephone", "narrow-band", "narrowband", "call center")
_TTS_TERMS = ("tts", "text-to-speech", "voice clone", "voice cloning")
_ALM_TERMS = ("alm", "audio language", "audio-language", "long-audio", "long window")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _has(text: str, *terms: str) -> bool:
    return any(t in text for t in terms)


def _suggest_quality_level(intent: IntentCategories) -> str:
    """Map the current quality settings → aggressiveness option id.

    Used as ``suggested_value`` on the quality gap so the rendered
    question pre-selects whichever level matches what
    ``infer_intent_from_prompt`` / ``apply_profile_prefills`` already
    settled. Mirrors the breakpoints in :func:`_quality_options`.
    """

    q = intent.quality
    if q.mos == FilterMode.ANNOTATE or q.sigmos == FilterMode.ANNOTATE:
        return "annotate"
    if q.mos == FilterMode.OFF and q.sigmos == FilterMode.OFF:
        # User mentioned quality but nothing concrete settled → suggest
        # the balanced default rather than "off".
        return "balanced"
    threshold = float(q.mos_threshold or 0.0)
    if threshold >= 4.0:
        return "strict"
    if threshold >= 3.4:
        return "balanced"
    if threshold > 0.0:
        return "light"
    return "balanced"


def _suggest_sample_rate(text: str, profile: DatasetCard | None) -> int | None:
    """Best-effort default for the sample-rate suggestion chip."""

    if _has(text, *_TTS_TERMS):
        return 24000
    if _has(text, *_PHONE_TERMS):
        return 8000
    if _has(text, "asr", "transcribe", "stt", "whisper"):
        return 16000
    if profile and profile.profile and profile.profile.sample_rates_hz:
        # Dominant rate in the dataset.
        rates = profile.profile.sample_rates_hz
        try:
            return int(max(rates.items(), key=lambda kv: kv[1])[0])
        except (ValueError, TypeError):
            return None
    return 16000


def analyze_gaps(
    intent: IntentCategories,
    prompt: str,
    profile: DatasetCard | None,
) -> list[Gap]:
    """Deterministic: return the *short* list of fields worth asking about.

    The rules are conservative — defaults silently win when:

    - the LLM extractor already filled the field, or
    - the prompt did not mention the family at all (no quality words → no
      quality question), or
    - the profile is unambiguous (single sample rate → suggest it as the
      default but still ask).
    """

    text = _norm(prompt)
    gaps: list[Gap] = []

    # ---- Essentials -------------------------------------------------------

    # Output sample rate is asked on every run, even when the heuristic
    # prefill or LLM extractor already wrote a value. The prefill is a
    # guess driven by keywords (TTS → 24 k, phone → 8 k, ASR → 16 k) or
    # the dataset profile's dominant rate; the user must confirm because
    # it changes resampling, model compatibility, and disk footprint.
    # The currently-settled value (if any) is the ``suggested_value`` so
    # the form pre-selects the right radio button.
    current_sr = intent.output.sample_rate
    if isinstance(current_sr, int):
        sr_suggest: Any = current_sr
    elif current_sr == "any":
        sr_suggest = "any"
    else:
        sr_suggest = _suggest_sample_rate(text, profile)
    sr_reason = (
        f"We picked {current_sr} Hz from your prompt / dataset — confirm or change."
        if isinstance(current_sr, int)
        else (
            "You asked us to keep input rates — confirm or override."
            if current_sr == "any"
            else (
                "The output sample rate sets the target for every file "
                "and decides whether we need a resample stage."
            )
        )
    )
    gaps.append(
        Gap(
            intent_path="output.sample_rate",
            category=(
                "essential_uncertain" if current_sr is not None else "essential_missing"
            ),
            reason=sr_reason,
            suggested_value=sr_suggest,
        )
    )

    if intent.output.audio_format is None:
        gaps.append(
            Gap(
                intent_path="output.audio_format",
                category="essential_missing",
                reason="Output container format (wav, flac, ogg, or manifest-only).",
                suggested_value="wav",
            )
        )

    # Resample-vs-drop is a conditional follow-up of sample-rate, but only
    # if the rate is already settled. If the rate gap is going to be asked
    # in this round, we attach this as that gap's follow-up below.
    rate_known = (
        intent.output.sample_rate is not None
        and intent.output.sample_rate != "any"
    )
    if rate_known and intent.output.resample_input is None:
        # The prompt may have said "drop mismatched"/"convert everything" —
        # in that case the extractor already set it. Otherwise ask.
        gaps.append(
            Gap(
                intent_path="output.resample_input",
                category="essential_missing",
                reason=(
                    "With a concrete sample rate, we need to know whether to "
                    "resample mismatched inputs or drop them."
                ),
                suggested_value=True,
            )
        )

    # ---- Conditional: duration -------------------------------------------
    # When the user uses duration wording we ALWAYS surface the
    # min/max question — even if the extractor or profile pre-fill
    # already wrote concrete values. The prefill is a guess (e.g. the
    # LLM mapped "short duration" → 10 s, or the dataset p05 → 1.3 s);
    # the user must confirm. Compare with the quality-aggressiveness
    # branch below — same pattern.
    seg = intent.segmentation
    duration_already_set = (
        seg.duration_min_sec is not None or seg.duration_max_sec is not None
    )
    duration_mentioned = _has(text, *_DURATION_TERMS)
    if duration_mentioned:
        # Pass currently-settled values as suggested defaults so the
        # form pre-selects them; the user can keep or override.
        gaps.append(
            Gap(
                intent_path="__duration_constraint__",
                category="prompt_mentioned",
                reason=(
                    "You mentioned clip / segment lengths."
                    if not duration_already_set
                    else (
                        "We picked min "
                        f"{seg.duration_min_sec}s / max "
                        f"{seg.duration_max_sec}s from your prompt and "
                        "dataset profile — confirm or change."
                    )
                ),
                suggested_value="yes" if duration_already_set else None,
                follow_ups=[
                    Gap(
                        intent_path="segmentation.duration_min_sec",
                        category="prompt_mentioned",
                        reason="Minimum clip length (seconds).",
                        suggested_value=seg.duration_min_sec or 2.0,
                    ),
                    Gap(
                        intent_path="segmentation.duration_max_sec",
                        category="prompt_mentioned",
                        reason="Maximum clip length (seconds).",
                        suggested_value=seg.duration_max_sec or 30.0,
                    ),
                ],
            )
        )

    # ---- Conditional: segmentation unit -----------------------------------
    # Ask only if the prompt is ambiguous about per-file vs per-segment.
    seg_ambiguous = (
        seg.output_unit == "original_files"
        and (
            _has(text, "clip", "segment", "split", "extract")
            or _has(text, *_ALM_TERMS)
        )
        and not _has(text, "original files", "per file", "per recording")
    )
    if seg_ambiguous:
        gaps.append(
            Gap(
                intent_path="segmentation.output_unit",
                category="prompt_mentioned",
                reason="You mentioned clips / segments — what should one output row represent?",
            )
        )

    # ---- Conditional: speakers --------------------------------------------

    speaker_mentioned = _has(text, *_SPEAKER_TERMS)
    if speaker_mentioned and intent.speakers.mode == FilterMode.OFF:
        gaps.append(
            Gap(
                intent_path="speakers.mode",
                category="prompt_mentioned",
                reason="You mentioned speakers / diarization — what should we do?",
            )
        )

    # ---- Conditional: quality / MOS / aggressiveness ----------------------
    # When the user uses any quality wording we always surface the
    # aggressiveness question — even if ``infer_intent_from_prompt`` /
    # ``apply_profile_prefills`` already prefilled a threshold. The
    # heuristic is a guess; this question is the user's say. The
    # currently-settled level is passed as ``suggested_value`` so the
    # form pre-selects it instead of forcing a re-pick.

    quality_mentioned = _has(text, *_QUALITY_TERMS)
    if quality_mentioned:
        gaps.append(
            Gap(
                intent_path="quality.mos",
                category="prompt_mentioned",
                reason=(
                    "You asked about audio quality / cleaning. Pick how "
                    "aggressive the filter should be — each level configures "
                    "UTMOS (naturalness) and SIGMOS (multi-axis quality) "
                    "thresholds together."
                ),
                suggested_value=_suggest_quality_level(intent),
            )
        )

    # ---- Conditional: transcripts -----------------------------------------

    transcript_mentioned = _has(text, *_TRANSCRIPT_TERMS)
    if transcript_mentioned and intent.text.transcript_source == "off":
        gaps.append(
            Gap(
                intent_path="text.transcript_source",
                category="prompt_mentioned",
                reason="You mentioned transcripts / ASR — generate, use existing, or skip?",
            )
        )

    # ---- TTS pivot questions ---------------------------------------------
    # "TTS dataset" is ambiguous about two things the user almost always
    # cares about — speaker fan-out and quality filtering — but neither
    # is explicitly said. The legacy clarifier used to *silently* enable
    # both, which produced wrong pipelines (see ticket: 332c0ecf06a981f6).
    # Surface them as questions so the user picks.
    tts_mentioned = _has(text, *_TTS_TERMS)
    if tts_mentioned:
        if not speaker_mentioned and not any(
            g.intent_path == "speakers.mode" for g in gaps
        ):
            gaps.append(Gap(
                intent_path="speakers.mode",
                category="prompt_mentioned",
                reason=(
                    "TTS datasets can be single-speaker (most common) or "
                    "per-speaker fan-out. Pick one — we won't guess."
                ),
                suggested_value="off",
            ))
        if not quality_mentioned and not any(
            g.intent_path == "quality.mos" for g in gaps
        ):
            gaps.append(Gap(
                intent_path="quality.mos",
                category="prompt_mentioned",
                reason=(
                    "TTS quality bar varies widely. Pick how aggressive "
                    "the MOS / SIGMOS filters should be (or skip filtering)."
                ),
                suggested_value=_suggest_quality_level(intent),
            ))

    # ---- Vague-prompt safety net -----------------------------------------
    # Short or generic prompts ("generate a dataset", "clean my data")
    # often leave critical fields unmentioned. Ask the user instead of
    # silently inheriting defaults — better one extra question than
    # producing a wrong pipeline.
    #
    # We skip a pivot if the field is *already* set to a concrete value,
    # because then the prompt-mentioned / heuristic prefill is at least
    # an explicit choice (or the extractor's). The point is to catch
    # the "nobody said anything, the pipeline used a hard-coded default"
    # case, not to second-guess every concrete value.
    word_count = sum(1 for w in text.split() if len(w) > 1)
    is_vague = word_count <= 12
    if is_vague:
        spk_mode_val = (
            intent.speakers.mode.value
            if hasattr(intent.speakers.mode, "value")
            else intent.speakers.mode
        )
        q_mos_val = (
            intent.quality.mos.value
            if hasattr(intent.quality.mos, "value")
            else intent.quality.mos
        )

        # Sample rate is handled by the unconditional essential gap above,
        # so it's intentionally omitted here to avoid duplicates.
        pivots: list[tuple[str, str, Any, bool]] = [
            (
                "speakers.mode",
                "How should we handle speakers — ignore, label them, filter by count, or split into per-speaker clips?",
                spk_mode_val,
                spk_mode_val == FilterMode.OFF.value or spk_mode_val == FilterMode.OFF,
            ),
            (
                "quality.mos",
                "Do you want quality filtering? If yes, pick how aggressive — otherwise we'll keep everything.",
                _suggest_quality_level(intent),
                q_mos_val == FilterMode.OFF.value or q_mos_val == FilterMode.OFF,
            ),
        ]
        existing = {g.intent_path for g in gaps}
        for path, reason, suggested, should_ask in pivots:
            if path in existing or not should_ask:
                continue
            gaps.append(Gap(
                intent_path=path,
                category="essential_uncertain",
                reason=reason,
                suggested_value=suggested,
            ))

    return gaps


# ----------------------------------------------------------------------------
# Inferred chips — what we already know
# ----------------------------------------------------------------------------


def _profile_dominant_sr(profile: DatasetCard | None) -> int | None:
    if profile is None or profile.profile is None:
        return None
    rates = profile.profile.sample_rates_hz
    if not rates:
        return None
    try:
        return int(max(rates.items(), key=lambda kv: kv[1])[0])
    except (ValueError, TypeError):
        return None


def build_inferred_chips(
    intent: IntentCategories,
    prompt: str,
    profile: DatasetCard | None,
) -> list[InferredChip]:
    """Convert the extractor's filled-in fields into user-visible chips."""

    text = _norm(prompt)
    chips: list[InferredChip] = []

    # output.sample_rate
    sr = intent.output.sample_rate
    if sr is not None:
        chips.append(
            InferredChip(
                intent_path="output.sample_rate",
                label=("Keep input rates" if sr == "any" else f"Sample rate: {sr} Hz"),
                value=sr,
                source=("prompt" if _has(text, "khz", "hz", "rate") or str(sr) in text else "profile" if profile else "default"),
                why=("From your prompt" if _has(text, "khz", "hz", "rate") else "Picked to match your goal"),
            )
        )

    # output.channels
    if intent.output.channels is not None:
        chips.append(
            InferredChip(
                intent_path="output.channels",
                label=f"Channels: {intent.output.channels}",
                value=intent.output.channels,
                source="prompt" if _has(text, "mono", "stereo", "channel") else "default",
                why=("From your prompt" if _has(text, "mono", "stereo", "channel") else "Default for this goal"),
            )
        )

    # output.audio_format
    if intent.output.audio_format is not None:
        chips.append(
            InferredChip(
                intent_path="output.audio_format",
                label=f"Format: {intent.output.audio_format}",
                value=intent.output.audio_format,
                source="prompt" if _has(text, ".wav", ".flac", ".ogg", "wav", "flac", "ogg") else "default",
                why="Default container format" if not _has(text, "wav", "flac", "ogg") else "From your prompt",
            )
        )

    # segmentation.output_unit (only chip if non-default)
    if intent.segmentation.output_unit != "original_files":
        chips.append(
            InferredChip(
                intent_path="segmentation.output_unit",
                label=f"Unit: {intent.segmentation.output_unit.replace('_', ' ')}",
                value=intent.segmentation.output_unit,
                source="prompt",
                why="Inferred from your wording",
            )
        )

    # duration
    seg = intent.segmentation
    if seg.duration_min_sec is not None or seg.duration_max_sec is not None:
        lo = seg.duration_min_sec
        hi = seg.duration_max_sec
        lo_s = f"{lo}s" if lo is not None else "any"
        hi_s = f"{hi}s" if hi is not None else "any"
        chips.append(
            InferredChip(
                intent_path="segmentation.duration_min_sec",
                label=f"Duration: {lo_s} – {hi_s}",
                value=[lo, hi],
                source="prompt",
                why="From your prompt",
            )
        )

    # quality.mos
    if intent.quality.mos != FilterMode.OFF:
        threshold = intent.quality.mos_threshold
        label = f"MOS filter ≥ {threshold}" if (intent.quality.mos == FilterMode.FILTER and threshold) else f"MOS {intent.quality.mos.value}"
        chips.append(
            InferredChip(
                intent_path="quality.mos",
                label=label,
                value=intent.quality.mos.value,
                source="prompt",
                why="Inferred from quality-related wording",
            )
        )

    if intent.quality.sigmos != FilterMode.OFF:
        chips.append(
            InferredChip(
                intent_path="quality.sigmos",
                label=f"SIGMOS {intent.quality.sigmos.value}",
                value=intent.quality.sigmos.value,
                source="prompt",
                why="Inferred from quality-related wording",
            )
        )

    if intent.quality.band != FilterMode.OFF and intent.quality.band_value:
        chips.append(
            InferredChip(
                intent_path="quality.band",
                label=f"Bandwidth: {intent.quality.band_value}",
                value=intent.quality.band_value,
                source="prompt",
                why="From narrow-band / wide-band wording",
            )
        )

    # speakers
    if intent.speakers.mode != FilterMode.OFF:
        chips.append(
            InferredChip(
                intent_path="speakers.mode",
                label=f"Speakers: {intent.speakers.mode.value}",
                value=intent.speakers.mode.value,
                source="prompt",
                why="From speaker / diarization wording",
            )
        )

    # text / transcript
    if intent.text.transcript_source != "off":
        chips.append(
            InferredChip(
                intent_path="text.transcript_source",
                label=f"Transcripts: {intent.text.transcript_source}",
                value=intent.text.transcript_source,
                source="prompt",
                why="From transcript / ASR wording",
            )
        )

    # commercial only
    if intent.policy.commercial_only:
        chips.append(
            InferredChip(
                intent_path="policy.commercial_only",
                label="Commercial-safe models only",
                value=True,
                source="prompt" if _has(text, "commercial", "production", "license") else "default",
                why="Inferred from commercial / production wording",
            )
        )

    # profile-derived chip: dominant sample rate when intent left it unset
    if sr is None and profile is not None:
        dom = _profile_dominant_sr(profile)
        if dom is not None:
            chips.append(
                InferredChip(
                    intent_path="__profile_sr__",
                    label=f"Dataset dominant rate: {dom} Hz",
                    value=dom,
                    source="profile",
                    why="Most common rate in the provided files",
                )
            )

    return chips


# ----------------------------------------------------------------------------
# Template fallback — deterministic question phrasing
# ----------------------------------------------------------------------------


def _sr_options(suggested: int | None) -> list[SmartOption]:
    presets = [
        ("16k", "16 kHz (ASR-friendly)", 16000),
        ("24k", "24 kHz (TTS / voice clone)", 24000),
        ("44k", "44.1 kHz", 44100),
        ("48k", "48 kHz (studio / broadcast)", 48000),
        ("8k", "8 kHz (telephone)", 8000),
    ]
    if suggested is not None and not any(v == suggested for _, _, v in presets):
        presets.insert(0, (f"{suggested}", f"{suggested} Hz (suggested)", suggested))
    opts: list[SmartOption] = [
        SmartOption(id=oid, label=label, value=val) for oid, label, val in presets
    ]
    opts.append(
        SmartOption(
            id="any",
            label="Keep input rates",
            description="Don't resample anything; pass through whatever the files already have.",
            value="any",
        )
    )
    opts.append(
        SmartOption(
            id="custom",
            label="Custom Hz…",
            description="Enter a sample rate in Hz.",
            is_freeform=True,
            freeform_kind="int_hz",
            freeform_placeholder="e.g. 22050",
        )
    )
    return opts


def _format_options() -> list[SmartOption]:
    return [
        SmartOption(id="wav", label="WAV", description="Uncompressed PCM, broadest support.", value="wav"),
        SmartOption(id="flac", label="FLAC", description="Lossless, smaller than WAV.", value="flac"),
        SmartOption(id="ogg", label="OGG / Vorbis", description="Lossy but compact.", value="ogg"),
    ]


def _resample_options() -> list[SmartOption]:
    return [
        SmartOption(
            id="resample",
            label="Convert every file to my chosen rate",
            description="Insert a resample stage so all outputs match the target.",
            value=True,
        ),
        SmartOption(
            id="drop",
            label="Only keep files that already match",
            description="Drop any file whose native rate differs — no resampling.",
            value=False,
        ),
    ]


def _seg_unit_options() -> list[SmartOption]:
    return [
        SmartOption(id="files", label="Original files", description="One row per input file.", value="original_files"),
        SmartOption(id="speech", label="Speech segments", description="Run VAD and emit one row per speech segment.", value="speech_segments"),
        SmartOption(id="windows", label="Long windows (ALM)", description="Pack into fixed-length chunks.", value="long_windows"),
        SmartOption(id="per_spk", label="Single-speaker clips", description="Diarize and emit one row per speaker.", value="single_speaker_clips"),
    ]


def _speakers_options() -> list[SmartOption]:
    # The ``filter`` option reveals ``q_speaker_count`` so the user
    # actually picks N — without that follow-up the selector silently
    # degrades FILTER to ANNOTATE (every row passes through). See run
    # c27f2786eaad1d75 for the regression this guards against.
    return [
        SmartOption(id="annotate", label="Tag rows with speaker info", description="Diarize and add num_speakers etc.", value="annotate"),
        SmartOption(
            id="filter",
            label="Keep only exactly-N speakers",
            description="Filter rows by speaker count.",
            value="filter",
            reveals=["q_speaker_count"],
        ),
        SmartOption(id="split", label="Split — one row per speaker", description="Speaker separation; fans out rows.", value="split"),
        SmartOption(id="off", label="Don't care / no speaker work", description="Skip the diarization stage entirely.", value="off"),
    ]


def _speaker_count_question(prefill: int | None = None) -> SmartQuestion:
    """Conditional follow-up of the speakers question (``filter`` mode).

    Picks the ``target_count`` so the selector's PBV gate actually
    drops files. The most common pick is ``1`` (TTS / voice cloning
    style file-level single-speaker filtering); meeting / interview
    use-cases use ``2`` or higher. Anything beyond ``4`` goes through
    the freeform ``custom`` option.
    """

    return SmartQuestion(
        id="q_speaker_count",
        intent_path="speakers.target_count",
        title="Keep files with how many speakers?",
        detail=(
            "We'll drop any file whose Sortformer-counted number of "
            "speakers doesn't match. For single-speaker TTS data pick 1."
        ),
        options=[
            SmartOption(id="one", label="Exactly 1 (single-speaker only)", value=1),
            SmartOption(id="two", label="Exactly 2 (dialog / interview)", value=2),
            SmartOption(id="three", label="Exactly 3", value=3),
            SmartOption(id="four", label="Exactly 4", value=4),
            SmartOption(
                id="custom",
                label="Custom count…",
                description="Enter a positive integer.",
                is_freeform=True,
                freeform_kind="speaker_count",
                freeform_placeholder="e.g. 5",
            ),
        ],
        prefill=prefill,
    )


def _quality_options() -> list[SmartOption]:
    """Aggressiveness ladder for the quality question.

    Each level applies a UTMOS + SIGMOS combination together via the
    option's ``apply`` dict, so picking one level is enough to set the
    whole quality block:

    - ``light``     — drop the worst (UTMOS ≥ 3.0, SIGMOS off)
    - ``balanced``  — clean speech (UTMOS ≥ 3.4 + SIGMOS ovrl/noise)
    - ``strict``    — studio / broadcast (UTMOS ≥ 4.0 + tight SIGMOS)
    - ``annotate``  — score but never drop
    - ``off``       — skip the whole quality block

    The selector still respects whatever the user picks (via
    ``apply_answers``); this question is purely about *which* preset to
    apply, not about hand-tuning each axis (that stays in the advanced
    form).
    """

    return [
        SmartOption(
            id="light",
            label="Light — drop only the worst (UTMOS ≥ 3.0)",
            description=(
                "Gentle cleaning. Removes obviously broken / clipped audio "
                "while keeping borderline-but-usable clips."
            ),
            value="filter",
            apply={
                "quality.mos": "filter",
                "quality.mos_threshold": 3.0,
                "quality.sigmos": "off",
            },
        ),
        SmartOption(
            id="balanced",
            label="Balanced — clean speech (UTMOS ≥ 3.4 + low-noise SIGMOS)",
            description=(
                "Recommended default. Drops unnatural-sounding clips and "
                "noisy ones; SIGMOS gates ovrl ≥ 3.5 and noise ≥ 4.0."
            ),
            value="filter",
            apply={
                "quality.mos": "filter",
                "quality.mos_threshold": 3.4,
                "quality.sigmos": "filter",
                "quality.sigmos_axes": ["ovrl", "noise"],
                "quality.sigmos_thresholds": {"ovrl": 3.5, "noise": 4.0},
            },
        ),
        SmartOption(
            id="strict",
            label="Strict — studio / broadcast (UTMOS ≥ 4.0 + tight SIGMOS)",
            description=(
                "Aggressive. Keeps only pristine clips suitable for "
                "production TTS / voice cloning; SIGMOS ovrl ≥ 4.0 and "
                "noise ≥ 4.0."
            ),
            value="filter",
            apply={
                "quality.mos": "filter",
                "quality.mos_threshold": 4.0,
                "quality.sigmos": "filter",
                "quality.sigmos_axes": ["ovrl", "noise"],
                "quality.sigmos_thresholds": {"ovrl": 4.0, "noise": 4.0},
            },
        ),
        SmartOption(
            id="annotate",
            label="Score every clip — keep them all",
            description=(
                "Run UTMOS + SIGMOS but don't drop anything. Scores land in "
                "the output manifest so you can filter offline later."
            ),
            value="annotate",
            apply={
                "quality.mos": "annotate",
                "quality.mos_threshold": 0.0,
                "quality.sigmos": "annotate",
                "quality.sigmos_axes": ["ovrl", "noise"],
            },
        ),
        SmartOption(
            id="off",
            label="Skip quality scoring entirely",
            description="No UTMOS / SIGMOS / Band stages will run.",
            value="off",
            apply={
                "quality.mos": "off",
                "quality.sigmos": "off",
                "quality.band": "off",
            },
        ),
    ]


def _transcript_options() -> list[SmartOption]:
    return [
        SmartOption(
            id="generate",
            label="Generate transcripts with ASR",
            description="Run NeMo / Whisper to produce text.",
            value="generate",
        ),
        SmartOption(
            id="generate_word",
            label="Generate + word-level timestamps",
            description="ASR with forced alignment.",
            value="generate",
            apply={"text.transcript_source": "generate", "text.word_timing": True},
        ),
        SmartOption(
            id="existing",
            label="Use existing transcripts from the manifest",
            description="Don't re-transcribe.",
            value="existing",
        ),
        SmartOption(
            id="off",
            label="No transcripts",
            description="Skip ASR.",
            value="off",
        ),
    ]


def _yes_no(yes_label: str, no_label: str, reveals_on_yes: list[str]) -> list[SmartOption]:
    return [
        SmartOption(id="no", label=no_label, value="no"),
        SmartOption(id="yes", label=yes_label, value="yes", reveals=reveals_on_yes),
    ]


def _duration_min_question(prefill: float | None = None) -> SmartQuestion:
    # When a value was inferred, surface it as the pre-selected radio
    # so the user sees what we picked and can keep or override.
    return SmartQuestion(
        id="q_duration_min",
        intent_path="segmentation.duration_min_sec",
        title="Minimum clip length (seconds)?",
        options=[
            SmartOption(id="1", label="1 s", value=1.0),
            SmartOption(id="2", label="2 s", value=2.0),
            SmartOption(id="5", label="5 s", value=5.0),
            SmartOption(
                id="custom",
                label="Custom…",
                is_freeform=True,
                freeform_kind="float_sec",
                freeform_placeholder="e.g. 0.5",
            ),
        ],
        prefill=prefill,
    )


def _duration_max_question(prefill: float | None = None) -> SmartQuestion:
    return SmartQuestion(
        id="q_duration_max",
        intent_path="segmentation.duration_max_sec",
        title="Maximum clip length (seconds)?",
        options=[
            SmartOption(id="10", label="10 s", value=10.0),
            SmartOption(id="30", label="30 s", value=30.0),
            SmartOption(id="60", label="60 s", value=60.0),
            SmartOption(id="120", label="120 s", value=120.0),
            SmartOption(
                id="custom",
                label="Custom…",
                is_freeform=True,
                freeform_kind="float_sec",
                freeform_placeholder="e.g. 45",
            ),
        ],
        prefill=prefill,
    )


def _template_question_for(gap: Gap, intent: IntentCategories) -> SmartQuestion | None:
    """Deterministic fallback phrasing for a single gap."""

    path = gap.intent_path

    if path == "output.sample_rate":
        return SmartQuestion(
            id="q_sample_rate",
            intent_path="output.sample_rate",
            title="What output sample rate do you want?",
            detail=gap.reason,
            options=_sr_options(gap.suggested_value if isinstance(gap.suggested_value, int) else None),
            prefill=gap.suggested_value,
        )

    if path == "output.audio_format":
        return SmartQuestion(
            id="q_audio_format",
            intent_path="output.audio_format",
            title="Output file format?",
            detail=gap.reason,
            options=_format_options(),
            prefill=gap.suggested_value or "wav",
        )

    if path == "output.resample_input":
        return SmartQuestion(
            id="q_resample",
            intent_path="output.resample_input",
            title="If a file doesn't already match your sample rate…",
            detail=gap.reason,
            options=_resample_options(),
            prefill=True,
        )

    if path == "segmentation.output_unit":
        return SmartQuestion(
            id="q_output_unit",
            intent_path="segmentation.output_unit",
            title="What does one output row represent?",
            detail=gap.reason,
            options=_seg_unit_options(),
        )

    if path == "speakers.mode":
        return SmartQuestion(
            id="q_speakers",
            intent_path="speakers.mode",
            title="How should we handle speakers?",
            detail=gap.reason,
            options=_speakers_options(),
            follow_ups=[_speaker_count_question(intent.speakers.target_count)],
        )

    if path == "quality.mos":
        suggested = gap.suggested_value if isinstance(gap.suggested_value, str) else None
        return SmartQuestion(
            id="q_quality",
            intent_path="quality.mos",
            title="How aggressively should we clean / filter the audio?",
            detail=gap.reason,
            options=_quality_options(),
            prefill=suggested,
        )

    if path == "text.transcript_source":
        return SmartQuestion(
            id="q_transcripts",
            intent_path="text.transcript_source",
            title="Do you want transcripts?",
            detail=gap.reason,
            options=_transcript_options(),
        )

    if path == "__duration_constraint__":
        # One yes/no, follow-ups for min and max. When the LLM
        # extractor or profile already wrote concrete values we
        # surface them as prefills on the follow-ups so the user
        # confirms instead of starting from a blank slate.
        seg = intent.segmentation
        prefill_min = seg.duration_min_sec
        prefill_max = seg.duration_max_sec
        return SmartQuestion(
            id="q_duration_enable",
            intent_path="__duration_constraint__",
            title="Should we filter by clip duration?",
            detail=gap.reason,
            options=_yes_no(
                yes_label="Yes — set min / max",
                no_label="No — keep any length",
                reveals_on_yes=["q_duration_min", "q_duration_max"],
            ),
            prefill=(
                "yes" if (prefill_min is not None or prefill_max is not None) else None
            ),
            follow_ups=[
                _duration_min_question(prefill_min),
                _duration_max_question(prefill_max),
            ],
        )

    logger.warning("smart_clarifier: no template for gap %s", path)
    return None


def template_questions(gaps: list[Gap], intent: IntentCategories) -> list[SmartQuestion]:
    """Deterministic fallback — used when the LLM is unavailable or errors."""

    out: list[SmartQuestion] = []
    for gap in gaps:
        q = _template_question_for(gap, intent)
        if q is not None:
            out.append(q)
    return out


# ----------------------------------------------------------------------------
# LLM composer — let the model phrase questions in the user's own words
# ----------------------------------------------------------------------------


_PROMPT_PATH = Path(__file__).parent / "nat" / "prompts" / "clarifier.md"


def _load_clarifier_prompt() -> str:
    if not _PROMPT_PATH.exists():
        msg = f"smart_clarifier system prompt missing at {_PROMPT_PATH}"
        raise FileNotFoundError(msg)
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _gap_payload(gap: Gap) -> dict[str, Any]:
    return {
        "intent_path": gap.intent_path,
        "category": gap.category,
        "reason": gap.reason,
        "suggested_value": gap.suggested_value,
        "follow_ups": [_gap_payload(fu) for fu in gap.follow_ups],
    }


def _llm_phrase_questions(
    prompt: str,
    intent: IntentCategories,
    gaps: list[Gap],
    llm: LLMClient,
    *,
    tier: str = "synth",
) -> list[SmartQuestion]:
    """Ask the LLM to phrase the gap list as ≤5 friendly questions.

    The LLM only chooses *titles, details, and option labels* — the set
    of intent paths and the option *values* are fixed by the template.
    This keeps the answers machine-parseable while letting the phrasing
    reference the user's own words.
    """

    if not gaps:
        return []

    template = template_questions(gaps, intent)
    if not template:
        return []

    sys_prompt = _load_clarifier_prompt()
    payload = {
        "user_prompt": prompt,
        "current_intent_summary": _intent_summary(intent),
        "gaps": [_gap_payload(g) for g in gaps],
        "template_questions": [q.model_dump(mode="json") for q in template],
        "constraints": {
            "max_top_level_questions": 5,
            "max_options_per_question": 5,
            "preserve_intent_paths": True,
            "preserve_option_values_and_apply": True,
            "preserve_follow_up_structure": True,
        },
    }
    messages = [
        Message("system", sys_prompt),
        Message("user", json.dumps(payload, indent=2)),
    ]
    raw = llm.chat_json(messages, tier=tier, purpose="smart_clarifier")
    questions_raw = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(questions_raw, list):
        msg = f"smart_clarifier: LLM did not return a 'questions' list (got: {type(raw).__name__})"
        raise ValueError(msg)

    parsed: list[SmartQuestion] = []
    template_by_id = {q.id: q for q in template}
    for raw_q in questions_raw:
        if not isinstance(raw_q, dict):
            continue
        qid = str(raw_q.get("id") or "")
        # Match against template by id; fall back by intent_path.
        anchor = template_by_id.get(qid) or _find_template_by_path(
            template, str(raw_q.get("intent_path") or "")
        )
        if anchor is None:
            continue
        try:
            parsed.append(_merge_llm_into_template(anchor, raw_q, template_by_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("smart_clarifier: dropping malformed LLM question %s: %s", qid, exc)
            continue

    # If the LLM dropped everything, fall back to the template.
    return parsed if parsed else template


def _find_template_by_path(template: list[SmartQuestion], path: str) -> SmartQuestion | None:
    for q in template:
        if q.intent_path == path:
            return q
    return None


def _merge_llm_into_template(
    anchor: SmartQuestion,
    raw_q: dict[str, Any],
    template_by_id: dict[str, SmartQuestion],
) -> SmartQuestion:
    """Take phrasing (title/detail/labels) from the LLM, keep values from the template.

    The template guarantees machine-correct intent paths and option values
    (and any ``apply`` side-effect dicts). The LLM only rewrites the
    user-facing strings.
    """

    title = str(raw_q.get("title") or anchor.title).strip() or anchor.title
    detail_raw = raw_q.get("detail")
    detail = str(detail_raw).strip() if isinstance(detail_raw, str) and detail_raw.strip() else anchor.detail

    raw_options = raw_q.get("options")
    new_options: list[SmartOption] = []
    if isinstance(raw_options, list):
        # Match each LLM option to one in the anchor by id (preferred) or label.
        anchor_by_id = {o.id: o for o in anchor.options}
        for raw_o in raw_options:
            if not isinstance(raw_o, dict):
                continue
            oid = str(raw_o.get("id") or "")
            base = anchor_by_id.get(oid)
            if base is None:
                continue
            new_label = str(raw_o.get("label") or base.label).strip() or base.label
            new_desc = raw_o.get("description")
            if isinstance(new_desc, str):
                new_desc = new_desc.strip() or None
            else:
                new_desc = base.description
            new_options.append(
                SmartOption(
                    id=base.id,
                    label=new_label,
                    description=new_desc,
                    value=base.value,
                    apply=base.apply,
                    is_freeform=base.is_freeform,
                    freeform_kind=base.freeform_kind,
                    freeform_placeholder=base.freeform_placeholder,
                    reveals=base.reveals,
                )
            )
    # If LLM produced nothing usable, keep template options unchanged.
    if not new_options:
        new_options = list(anchor.options)
    else:
        # Make sure we didn't drop any anchor option that has 'reveals'
        # — losing those would silently disable conditional follow-ups.
        seen = {o.id for o in new_options}
        for base in anchor.options:
            if base.id not in seen and base.reveals:
                new_options.append(base)

    return SmartQuestion(
        id=anchor.id,
        intent_path=anchor.intent_path,
        title=title,
        detail=detail,
        options=new_options,
        follow_ups=list(anchor.follow_ups),
        visible_when=anchor.visible_when,
        prefill=anchor.prefill,
        is_freeform_root=anchor.is_freeform_root,
        freeform_kind=anchor.freeform_kind,
        freeform_placeholder=anchor.freeform_placeholder,
    )


def _intent_summary(intent: IntentCategories) -> dict[str, Any]:
    """Compact, LLM-readable view of what's already settled."""

    def _nonempty(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if v not in (None, "", [], {}, "off")}

    return {
        "output": _nonempty(intent.output.model_dump(mode="json")),
        "segmentation": _nonempty(intent.segmentation.model_dump(mode="json")),
        "quality": _nonempty(intent.quality.model_dump(mode="json")),
        "speakers": _nonempty(intent.speakers.model_dump(mode="json")),
        "text": _nonempty(intent.text.model_dump(mode="json")),
        "policy": _nonempty(intent.policy.model_dump(mode="json")),
    }


# ----------------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------------


def compose_smart_form(
    prompt: str,
    intent: IntentCategories,
    *,
    profile: DatasetCard | None = None,
    llm: LLMClient | None = None,
    tier: str = "synth",
) -> SmartForm:
    """Build the adaptive smart form for the web UI.

    ``intent`` should already be the output of ``extract_intent``. We then
    apply the prompt-regex + profile heuristics from the legacy clarifier
    as a *safety net* — the LLM extractor occasionally misses things the
    regex catches ("24 kHz", "between 2 and 60 seconds", etc.). Anything
    the heuristics fill in lowers the gap count and shows up as an
    inferred chip with ``source="prompt"`` or ``"profile"``.
    """

    enriched, prompt_assumptions = infer_intent_from_prompt(prompt, intent)
    enriched, profile_assumptions = apply_profile_prefills(enriched, profile)

    gaps = analyze_gaps(enriched, prompt, profile)

    questions: list[SmartQuestion]
    if llm is not None and gaps:
        try:
            questions = _llm_phrase_questions(prompt, enriched, gaps, llm, tier=tier)
        except Exception as exc:  # noqa: BLE001
            logger.warning("smart_clarifier: LLM composer failed (%s); using template", exc)
            questions = template_questions(gaps, enriched)
    else:
        questions = template_questions(gaps, enriched)

    inferred = build_inferred_chips(enriched, prompt, profile)
    advanced = build_clarification_form(prompt, enriched, profile=profile)

    assumptions: list[str] = []
    if inferred:
        assumptions.append(
            f"Extracted {len(inferred)} settings from your prompt + dataset profile."
        )
    if not gaps:
        assumptions.append("No clarification needed — ready to build.")
    # Surface the heuristic provenance notes too (deduped, capped) so the
    # UI can show them in the inferred-card expander.
    extra = list(dict.fromkeys(prompt_assumptions + profile_assumptions))[:8]
    assumptions.extend(extra)

    return SmartForm(
        inferred=inferred,
        questions=questions,
        assumptions=assumptions,
        profile_summary=_profile_summary(profile),
        advanced_form=advanced,
        intent=enriched,
    )


def _profile_summary(profile: DatasetCard | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    prof = profile.profile
    return {
        "total_files": prof.total_files,
        "decodable_files": prof.decodable_files,
        "decode_failure_rate": prof.decode_failure_rate,
        "sample_rates_hz": prof.sample_rates_hz,
        "channel_distribution": prof.channel_distribution,
        "duration_p05_sec": prof.duration_p05_sec,
        "duration_p50_sec": prof.duration_p50_sec,
        "duration_p95_sec": prof.duration_p95_sec,
        "total_duration_hours": prof.total_duration_hours,
    }


__all__ = [
    "Gap",
    "InferredChip",
    "SmartForm",
    "SmartOption",
    "SmartQuestion",
    "analyze_gaps",
    "apply_answers",
    "build_inferred_chips",
    "compose_smart_form",
    "template_questions",
]
