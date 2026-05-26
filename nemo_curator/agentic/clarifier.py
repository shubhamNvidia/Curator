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
"""Ingredient-picker clarification.

This module replaces the previous goal-anchored clarifier with a flat,
ingredient-picker form. It is documented in detail in ``INTENT_V2.md``:

1. The user submits a prompt.
2. ``infer_intent_from_prompt`` extracts what it can from the prompt's
   wording (no goal taxonomy — each ingredient is inferred independently).
3. ``apply_profile_prefills`` reads the dataset's :class:`DatasetCard`
   profile and pre-fills more ingredients (with ``source="profile"``).
4. ``build_clarification_form`` returns one :class:`ClarificationForm`
   containing every ingredient question. Visibility is computed
   server-side from the current intent so the client only sees relevant
   follow-ups.
5. ``apply_answers`` merges the user's picks back into the intent.

There is no clarification *loop*: the server returns the whole form
once, the user adjusts, and ``apply_answers`` finalizes the intent.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_curator.agentic.cards import DatasetCard
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    SigmosAxis,
)

# ----------------------------------------------------------------------------
# Form data model
# ----------------------------------------------------------------------------


PrefillSource = Literal["prompt", "profile", "default", "answer"]
FreeformKind = Literal["int_hz", "int", "float", "string", "duration_range", "speaker_count"]


class Prefill(BaseModel):
    """What the form arrived already filled with, and why."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    source: PrefillSource
    reason: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class Option(BaseModel):
    """One pickable answer.

    Either ``value`` is a literal (the option is a fixed pick) or
    ``is_freeform=True`` and the client must collect the user's typed
    value before submission. ``apply`` is the intent path → value(s) the
    server should write when the option is selected; defaults to writing
    ``value`` to the question's ``intent_path``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    description: str = ""
    value: Any = None
    apply: dict[str, Any] = Field(default_factory=dict)
    is_freeform: bool = False
    freeform_kind: FreeformKind | None = None
    freeform_placeholder: str | None = None
    notes: list[str] = Field(default_factory=list)


class Question(BaseModel):
    """One ingredient question with its options."""

    model_config = ConfigDict(extra="forbid")

    id: str
    intent_path: str
    title: str
    why: str
    options: list[Option]
    prefill: Prefill | None = None
    visible: bool = True
    visible_when: str | None = None


class Section(BaseModel):
    """Grouped questions surfacing one ingredient family."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    summary: str
    questions: list[Question]


class ClarificationForm(BaseModel):
    """One-shot ingredient form returned to the UI."""

    model_config = ConfigDict(extra="forbid")

    sections: list[Section]
    intent: IntentCategories
    profile_summary: dict[str, Any] | None = None
    assumptions: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Prompt-driven heuristic inference
# ----------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _has(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text))


_TTS_TERMS = ("tts", "text to speech", "text-to-speech", "voice clone", "voice cloning")
_PHONE_TERMS = ("phone call", "phone-call", "telephone", "call center", "call-center", "narrowband", "narrow-band")
_TRANSCRIPT_TERMS = ("transcript", "transcribe", "subtitles", "captions", "stt")
_DIARIZE_TERMS = ("speaker label", "who said", "diariz", "meeting", "interview")
_ALM_TERMS = ("audio-language", "audio language", "alm", "long-audio", "long audio chunk")
_SPEECH_PRESENCE_TERMS = (
    "at least speech", "atleast speech", "speech is there", "must have speech",
    "drop empty", "drop silent", "remove silent", "no silence", "without silence",
    "non-silent", "non silent", "skip empty", "skip silent", "remove empty audio",
    "no empty audio", "filter empty", "filter silent", "speech-only", "speech only",
)
_HIGH_QUALITY_TERMS = (
    "high quality", "studio", "broadcast", "clean audio", "clean speech", "pristine",
    "best quality", "premium",
)
_BALANCED_QUALITY_TERMS = ("good quality", "decent quality", "clean ", "non-noisy", "denoised")
_NOISY_KEEP_TERMS = ("keep noisy", "noisy", "low quality")
_SPLIT_SPEAKERS_TERMS = ("one speaker per", "single speaker", "per speaker", "separate speakers")
_SR_PATTERN = re.compile(r"\b(8|16|22\.05|24|32|44\.1|48)\s*k(?:hz)?\b")
_SR_HZ_PATTERN = re.compile(r"\b(8000|16000|22050|24000|32000|44100|48000)\b")


def infer_intent_from_prompt(prompt: str, intent: IntentCategories) -> tuple[IntentCategories, list[str]]:
    """Update ``intent`` in-place-style with prompt-driven prefills.

    Returns the updated intent and a list of human-readable assumption
    sentences. Each inference is independent — there is no goal anchor.
    """

    text = _norm(prompt)
    data = intent.model_dump(mode="json")
    assumptions: list[str] = []

    # --- Sample rate from explicit "X kHz" / "Xkhz" in prompt -----------
    if data["output"].get("sample_rate") in (None, "any"):
        sr = _parse_sample_rate(text)
        if sr is not None:
            data["output"]["sample_rate"] = sr
            data["output"]["resample_input"] = True
            assumptions.append(f"Prompt mentions ~{sr} Hz; pre-filling output sample rate.")

    # --- Channels ------------------------------------------------------
    if data["output"].get("channels") is None:
        if _has(text, "mono"):
            data["output"]["channels"] = "mono"
            assumptions.append("Prompt mentions mono; pre-filling channels=mono.")
        elif _has(text, "stereo"):
            data["output"]["channels"] = "stereo"
            assumptions.append("Prompt mentions stereo; pre-filling channels=stereo.")

    # --- Output unit ---------------------------------------------------
    seg = data["segmentation"]
    if _has(text, *_ALM_TERMS):
        seg["output_unit"] = "long_windows"
        if seg.get("long_window_sec") is None:
            seg["long_window_sec"] = 120.0
        data["text"]["transcript_source"] = "generate"
        data["text"]["word_timing"] = True
        assumptions.append("Prompt mentions ALM / long-audio; pre-filling long-window packaging + ASR alignment.")
    elif _has(text, *_SPLIT_SPEAKERS_TERMS):
        seg["output_unit"] = "single_speaker_clips"
        data["speakers"]["mode"] = FilterMode.SPLIT.value
        assumptions.append("Prompt asks for single-speaker clips; pre-filling segmentation + speaker SPLIT.")
    elif _has(text, "speech clip", "speech clips", "voice clip", "voice clips", "segment", "segments"):
        seg["output_unit"] = "speech_segments"
        assumptions.append("Prompt mentions speech clips/segments; pre-filling output_unit=speech_segments.")

    # --- Duration limits from "X-Y seconds" ---------------------------
    rng = _parse_duration_range(text)
    if rng is not None and seg.get("duration_min_sec") is None and seg.get("duration_max_sec") is None:
        lo, hi = rng
        seg["duration_min_sec"] = lo
        seg["duration_max_sec"] = hi
        if seg["output_unit"] == "original_files":
            seg["output_unit"] = "speech_segments"
            assumptions.append(
                f"Prompt gives a duration range ({lo}-{hi}s); switching output_unit to speech_segments and enabling VAD."
            )
        else:
            assumptions.append(f"Prompt gives a duration range ({lo}-{hi}s); pre-filling segmentation limits.")

    # --- Speech presence ----------------------------------------------
    if _has(text, *_SPEECH_PRESENCE_TERMS):
        if seg["output_unit"] == "original_files" and seg["speech_policy"] == FilterMode.OFF.value:
            seg["speech_policy"] = FilterMode.FILTER.value
            assumptions.append(
                "Prompt asks to drop empty audio; enabling speech_policy=FILTER (VAD runs in nested mode)."
            )

    # --- Quality -------------------------------------------------------
    q = data["quality"]
    if _has(text, *_HIGH_QUALITY_TERMS) and q["mos"] == FilterMode.OFF.value:
        q["mos"] = FilterMode.FILTER.value
        q["mos_threshold"] = 4.0
        q["sigmos"] = FilterMode.FILTER.value
        q["sigmos_axes"] = ["ovrl", "noise"]
        q["sigmos_thresholds"] = {"ovrl": 4.0, "noise": 4.0}
        assumptions.append("Prompt asks for studio/broadcast/high-quality audio; enabling strict MOS + SIGMOS filters.")
    elif _has(text, *_BALANCED_QUALITY_TERMS) and q["mos"] == FilterMode.OFF.value:
        q["mos"] = FilterMode.FILTER.value
        q["mos_threshold"] = 3.4
        q["sigmos"] = FilterMode.FILTER.value
        q["sigmos_axes"] = ["ovrl", "noise"]
        q["sigmos_thresholds"] = {"ovrl": 3.5, "noise": 4.0}
        assumptions.append("Prompt asks for clean/decent audio; enabling balanced MOS + SIGMOS filters.")

    # --- Bandwidth (phone vs full-band) ---------------------------------
    if _has(text, *_PHONE_TERMS):
        if q["band"] == FilterMode.OFF.value:
            q["band"] = FilterMode.FILTER.value
            q["band_value"] = "narrow_band"
            assumptions.append("Prompt mentions phone/telephone; enabling BandFilter=narrow_band.")
        if data["output"].get("sample_rate") in (None, "any"):
            data["output"]["sample_rate"] = 16000
            data["output"]["resample_input"] = True
            assumptions.append("Phone-style audio → output sample rate defaulted to 16 kHz.")

    # --- Speakers ------------------------------------------------------
    if _has(text, *_DIARIZE_TERMS) and data["speakers"]["mode"] == FilterMode.OFF.value:
        data["speakers"]["mode"] = FilterMode.ANNOTATE.value
        assumptions.append("Prompt mentions speaker labels / diarization; enabling speaker annotate.")

    # --- Text / ASR ----------------------------------------------------
    if _has(text, *_TRANSCRIPT_TERMS) and data["text"]["transcript_source"] == "off":
        data["text"]["transcript_source"] = "generate"
        if _has(text, "word", "timestamp", "timing", "align"):
            data["text"]["word_timing"] = True
        assumptions.append("Prompt mentions transcripts/ASR; enabling text.transcript_source=generate.")

    # --- TTS quick-pack ------------------------------------------------
    # Side-effects are independent so they layer on top of any earlier
    # high-quality / balanced detection: TTS flips speakers→SPLIT and the
    # output SR default to 24 kHz, even when the quality block has
    # already set MOS=FILTER from a "clean" word.
    if _has(text, *_TTS_TERMS):
        changed_anything = False
        if data["quality"]["mos"] == FilterMode.OFF.value:
            data["quality"]["mos"] = FilterMode.FILTER.value
            data["quality"]["mos_threshold"] = 3.4
            changed_anything = True
        if data["speakers"]["mode"] == FilterMode.OFF.value:
            data["speakers"]["mode"] = FilterMode.SPLIT.value
            changed_anything = True
        if data["output"].get("sample_rate") in (None, "any"):
            data["output"]["sample_rate"] = 24000
            data["output"]["resample_input"] = True
            changed_anything = True
        if changed_anything:
            assumptions.append(
                "Prompt mentions TTS / voice cloning; pre-filling speakers=SPLIT, mos=FILTER, output SR=24 kHz."
            )

    data["raw_prompt"] = data.get("raw_prompt") or prompt
    if assumptions:
        notes = list(data.get("notes") or [])
        notes.extend(f"prompt_inference: {a}" for a in assumptions)
        data["notes"] = _dedupe(notes)

    return IntentCategories(**data), assumptions


def apply_profile_prefills(
    intent: IntentCategories,
    profile: DatasetCard | None,
) -> tuple[IntentCategories, list[str]]:
    """Use the dataset profile to fill in still-unanswered ingredients."""

    if profile is None or profile.profile is None:
        return intent, []

    prof = profile.profile
    data = intent.model_dump(mode="json")
    assumptions: list[str] = []

    if data["output"].get("sample_rate") is None and prof.sample_rates_hz:
        keys = list(prof.sample_rates_hz.keys())
        if len(keys) == 1:
            try:
                sr = int(keys[0])
            except ValueError:
                sr = None
            if sr in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
                data["output"]["sample_rate"] = sr
                data["output"]["resample_input"] = False
                assumptions.append(
                    f"Dataset profile shows every file is {sr} Hz; pre-filling output SR with no resample."
                )

    if data["output"].get("channels") is None and prof.channel_distribution:
        keys = list(prof.channel_distribution.keys())
        if len(keys) == 1 and keys[0] in {"1", "2"}:
            data["output"]["channels"] = "mono" if keys[0] == "1" else "stereo"
            assumptions.append(
                f"Dataset profile shows every file is {data['output']['channels']}; pre-filling channels."
            )

    if (
        data["segmentation"]["output_unit"] == "speech_segments"
        and data["segmentation"]["duration_min_sec"] is None
        and prof.duration_p05_sec is not None
    ):
        data["segmentation"]["duration_min_sec"] = max(0.5, float(prof.duration_p05_sec) * 0.5)
        assumptions.append(
            f"Dataset p05 duration ≈ {prof.duration_p05_sec:.1f}s; pre-filling min duration as a permissive floor."
        )

    if assumptions:
        notes = list(data.get("notes") or [])
        notes.extend(f"profile_inference: {a}" for a in assumptions)
        data["notes"] = _dedupe(notes)

    return IntentCategories(**data), assumptions


# ----------------------------------------------------------------------------
# Form construction
# ----------------------------------------------------------------------------


def build_clarification_form(
    prompt: str,
    intent: IntentCategories,
    *,
    profile: DatasetCard | None = None,
    answered: dict[str, Any] | None = None,
) -> ClarificationForm:
    """Return the full ingredient form, pre-filled and visibility-resolved."""

    intent_with_prompt, prompt_assumptions = infer_intent_from_prompt(prompt, intent)
    intent_with_profile, profile_assumptions = apply_profile_prefills(intent_with_prompt, profile)
    if answered:
        intent_with_profile = apply_answers(intent_with_profile, answered)
    intent_normalized = intent_with_profile

    sections = [
        _output_section(intent_normalized, profile),
        _segmentation_section(intent_normalized),
        _quality_section(intent_normalized),
        _annotations_section(intent_normalized),
        _policy_section(intent_normalized),
    ]
    assumptions = _dedupe(prompt_assumptions + profile_assumptions)
    return ClarificationForm(
        sections=sections,
        intent=intent_normalized,
        profile_summary=_profile_summary(profile),
        assumptions=assumptions,
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


# ----------------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------------


def _output_section(intent: IntentCategories, profile: DatasetCard | None) -> Section:
    sr = intent.output.sample_rate
    prefill_source = _source_for_sample_rate(intent, profile)
    sample_rate_options = [
        Option(id="8k", label="8 kHz (phone)", value=8000),
        Option(id="16k", label="16 kHz (ASR / general)", value=16000),
        Option(id="22k", label="22.05 kHz", value=22050),
        Option(id="24k", label="24 kHz (voice cloning)", value=24000),
        Option(id="32k", label="32 kHz", value=32000),
        Option(id="44k", label="44.1 kHz", value=44100),
        Option(id="48k", label="48 kHz (high quality)", value=48000),
        Option(id="any", label="Keep input rates", value="any"),
        Option(
            id="custom",
            label="Write your own kHz",
            description="Enter a custom sample rate in Hz.",
            is_freeform=True,
            freeform_kind="int_hz",
            freeform_placeholder="e.g. 22050",
        ),
    ]

    sr_question = Question(
        id="output.sample_rate",
        intent_path="output.sample_rate",
        title="What output sample rate?",
        why=(
            "Sets the target sample rate of every output file. Picking a concrete value "
            "unlocks the resample-vs-drop question below."
        ),
        options=sample_rate_options,
        prefill=Prefill(
            value=sr,
            source=prefill_source,
            reason=_explain_sample_rate(intent, profile),
        ) if sr is not None else None,
    )

    resample_question = Question(
        id="output.resample_input",
        intent_path="output.resample_input",
        title="Resample input files to this rate?",
        why=(
            "If you say yes, every input file is converted to the target rate. "
            "If no, files that don't match are dropped — useful when you want to "
            "preserve original encodings."
        ),
        options=[
            Option(id="yes", label="Yes — convert all files", value=True),
            Option(id="no", label="No — drop mismatched files", value=False),
        ],
        prefill=Prefill(
            value=intent.output.resample_input,
            source="default" if intent.output.resample_input is None else "prompt",
            reason="Defaults to converting; only relevant when sample_rate is a concrete number.",
        ),
        visible=isinstance(sr, int),
        visible_when="output.sample_rate is a concrete Hz value",
    )

    channels_question = Question(
        id="output.channels",
        intent_path="output.channels",
        title="Channel layout?",
        why="Mono is required for most analytical stages. Stereo is preserved as-is when chosen.",
        options=[
            Option(id="mono", label="Mono", value="mono"),
            Option(id="stereo", label="Stereo", value="stereo"),
            Option(id="any", label="Match input", value="any"),
            Option(id="skip", label="Skip — no channel constraint", value=None),
        ],
        prefill=Prefill(
            value=intent.output.channels,
            source=_source_for_channels(intent, profile),
            reason=_explain_channels(intent, profile),
        ) if intent.output.channels is not None else None,
        visible=_should_show_channels(profile, intent),
        visible_when="dataset profile is mixed-channel OR prompt does not pick a channel layout",
    )

    format_question = Question(
        id="output.audio_format",
        intent_path="output.audio_format",
        title="Output audio file format?",
        why="The container the writer uses for materialized clips.",
        options=[
            Option(id="wav", label="WAV", value="wav"),
            Option(id="flac", label="FLAC", value="flac"),
            Option(id="ogg", label="OGG", value="ogg"),
            Option(id="skip", label="Skip — manifest only", value=None),
        ],
        prefill=Prefill(
            value=intent.output.audio_format,
            source="default",
            reason="Defaults to WAV.",
        ),
    )

    return Section(
        id="output",
        title="Output format",
        summary="How each output file should look.",
        questions=[sr_question, resample_question, channels_question, format_question],
    )


def _segmentation_section(intent: IntentCategories) -> Section:
    unit_q = Question(
        id="segmentation.output_unit",
        intent_path="segmentation.output_unit",
        title="What does one output row represent?",
        why="Drives whether VAD, long-window split, or speaker split is added.",
        options=[
            Option(id="original_files", label="One row per original file", value="original_files"),
            Option(id="speech_segments", label="One row per VAD speech segment", value="speech_segments"),
            Option(id="long_windows", label="Long-audio windows (ALM-style)", value="long_windows"),
            Option(id="single_speaker_clips", label="Single-speaker clips", value="single_speaker_clips"),
        ],
        prefill=Prefill(
            value=intent.segmentation.output_unit,
            source="prompt" if intent.segmentation.output_unit != "original_files" else "default",
            reason="Inferred from the prompt or kept at the default.",
        ),
    )

    duration_q = Question(
        id="segmentation.duration",
        intent_path="segmentation.duration_min_sec",
        title="Clip duration limits?",
        why="Tighter ranges drop ultra-short and ultra-long clips at the VAD stage.",
        options=[
            Option(id="tts", label="2–60 s (TTS default)", value=None, apply={
                "segmentation.duration_min_sec": 2.0,
                "segmentation.duration_max_sec": 60.0,
            }),
            Option(id="asr", label="2–30 s (ASR / call default)", value=None, apply={
                "segmentation.duration_min_sec": 2.0,
                "segmentation.duration_max_sec": 30.0,
            }),
            Option(id="long", label="5–120 s (long chunks)", value=None, apply={
                "segmentation.duration_min_sec": 5.0,
                "segmentation.duration_max_sec": 120.0,
            }),
            Option(id="none", label="No limit", value=None, apply={
                "segmentation.duration_min_sec": None,
                "segmentation.duration_max_sec": None,
            }),
            Option(
                id="custom",
                label="Write your own min,max in seconds",
                description="Enter '<min>,<max>' (e.g. '3,45').",
                is_freeform=True,
                freeform_kind="duration_range",
                freeform_placeholder="3,45",
            ),
        ],
        prefill=_duration_prefill(intent),
        visible=intent.segmentation.output_unit != "original_files",
        visible_when="output_unit != original_files",
    )

    speech_policy_q = Question(
        id="segmentation.speech_policy",
        intent_path="segmentation.speech_policy",
        title="Speech-presence policy on whole files?",
        why=(
            "When the row schema is the original file, VAD can either tag rows with "
            "speech segments or attempt to drop rows that have no speech at all."
        ),
        options=[
            Option(id="off", label="Off — don't run VAD", value=FilterMode.OFF.value),
            Option(id="annotate", label="Annotate only — add speech segments to manifest", value=FilterMode.ANNOTATE.value),
            Option(id="filter", label="Filter — drop files with no detected speech", value=FilterMode.FILTER.value),
        ],
        prefill=Prefill(
            value=intent.segmentation.speech_policy.value,
            source="prompt" if intent.segmentation.speech_policy != FilterMode.OFF else "default",
            reason="Inferred from prompt or kept off by default.",
        ),
        visible=intent.segmentation.output_unit == "original_files",
        visible_when="output_unit == original_files",
    )

    long_window_q = Question(
        id="segmentation.long_window_sec",
        intent_path="segmentation.long_window_sec",
        title="Long-audio window length?",
        why="Required for ALM-style packaging.",
        options=[
            Option(id="30s", label="30 s", value=30.0),
            Option(id="60s", label="60 s", value=60.0),
            Option(id="120s", label="120 s", value=120.0),
            Option(
                id="custom",
                label="Write your own seconds",
                is_freeform=True,
                freeform_kind="float",
                freeform_placeholder="e.g. 45",
            ),
        ],
        prefill=Prefill(
            value=intent.segmentation.long_window_sec,
            source="default" if intent.segmentation.long_window_sec is None else "prompt",
            reason="120 s default for ALM packaging.",
        ),
        visible=intent.segmentation.output_unit == "long_windows",
        visible_when="output_unit == long_windows",
    )

    return Section(
        id="segmentation",
        title="Segmentation & speech",
        summary="What one row represents, duration limits, and the speech-presence policy.",
        questions=[unit_q, duration_q, speech_policy_q, long_window_q],
    )


def _quality_section(intent: IntentCategories) -> Section:
    mos_q = Question(
        id="quality.mos",
        intent_path="quality.mos",
        title="Naturalness (UTMOS) gate?",
        why="UTMOS is a calibrated naturalness MOS estimator. Annotate-only writes the score; filter drops below the threshold.",
        options=[
            Option(id="off", label="Off", value=FilterMode.OFF.value),
            Option(
                id="annotate",
                label="Annotate only — write utmos_mos to manifest",
                value=FilterMode.ANNOTATE.value,
            ),
            Option(
                id="strict",
                label="Filter — strict (≥4.0)",
                value=FilterMode.FILTER.value,
                apply={"quality.mos_threshold": 4.0},
            ),
            Option(
                id="balanced",
                label="Filter — balanced (≥3.4)",
                value=FilterMode.FILTER.value,
                apply={"quality.mos_threshold": 3.4},
            ),
            Option(
                id="loose",
                label="Filter — loose (≥3.0)",
                value=FilterMode.FILTER.value,
                apply={"quality.mos_threshold": 3.0},
            ),
            Option(
                id="custom",
                label="Filter — write your own threshold",
                is_freeform=True,
                freeform_kind="float",
                freeform_placeholder="e.g. 3.7",
                apply={"quality.mos": FilterMode.FILTER.value},
            ),
        ],
        prefill=Prefill(
            value=intent.quality.mos.value,
            source="prompt" if intent.quality.mos != FilterMode.OFF else "default",
            reason="UTMOS gate state.",
        ),
    )

    sigmos_q = Question(
        id="quality.sigmos",
        intent_path="quality.sigmos",
        title="Multi-axis quality (SIGMOS) gate?",
        why="SIGMOS provides per-axis scores (overall, noise, signal, coloration, discontinuity, loudness, reverb).",
        options=[
            Option(id="off", label="Off", value=FilterMode.OFF.value),
            Option(
                id="annotate_overall_noise",
                label="Annotate — overall + noise",
                value=FilterMode.ANNOTATE.value,
                apply={"quality.sigmos_axes": ["ovrl", "noise"]},
            ),
            Option(
                id="annotate_all",
                label="Annotate — every axis",
                value=FilterMode.ANNOTATE.value,
                apply={
                    "quality.sigmos_axes": ["ovrl", "noise", "sig", "col", "disc", "loud", "reverb"],
                },
            ),
            Option(
                id="filter_balanced",
                label="Filter — overall + noise (balanced)",
                value=FilterMode.FILTER.value,
                apply={
                    "quality.sigmos_axes": ["ovrl", "noise"],
                    "quality.sigmos_thresholds": {"ovrl": 3.5, "noise": 4.0},
                },
            ),
            Option(
                id="filter_broadcast",
                label="Filter — broadcast (overall + noise + col + loud)",
                value=FilterMode.FILTER.value,
                apply={
                    "quality.sigmos_axes": ["ovrl", "noise", "col", "loud"],
                    "quality.sigmos_thresholds": {
                        "ovrl": 4.0, "noise": 4.0, "col": 4.0, "loud": 3.5,
                    },
                },
            ),
            Option(
                id="filter_studio",
                label="Filter — studio-strict (all axes)",
                value=FilterMode.FILTER.value,
                apply={
                    "quality.sigmos_axes": ["ovrl", "noise", "sig", "col", "disc", "loud", "reverb"],
                    "quality.sigmos_thresholds": {
                        "ovrl": 4.0, "noise": 4.5, "sig": 4.0, "col": 4.0,
                        "disc": 4.0, "loud": 3.5, "reverb": 4.0,
                    },
                },
            ),
        ],
        prefill=Prefill(
            value=intent.quality.sigmos.value,
            source="prompt" if intent.quality.sigmos != FilterMode.OFF else "default",
            reason="SIGMOS gate state.",
        ),
        visible=intent.quality.mos != FilterMode.OFF or intent.quality.sigmos != FilterMode.OFF,
        visible_when="quality.mos != OFF (or sigmos already set)",
    )

    band_q = Question(
        id="quality.band",
        intent_path="quality.band",
        title="Bandwidth filter?",
        why=(
            "Drops clips that don't match the chosen bandwidth class. "
            "Note: BandFilter has no annotate-only mode in the current catalog."
        ),
        options=[
            Option(id="off", label="Off", value=FilterMode.OFF.value),
            Option(
                id="narrow",
                label="Filter — narrow-band only (phone)",
                value=FilterMode.FILTER.value,
                apply={"quality.band_value": "narrow_band"},
            ),
            Option(
                id="full",
                label="Filter — full-band only (studio)",
                value=FilterMode.FILTER.value,
                apply={"quality.band_value": "full_band"},
            ),
        ],
        prefill=Prefill(
            value=intent.quality.band.value,
            source="prompt" if intent.quality.band != FilterMode.OFF else "default",
            reason="Bandwidth filter state.",
        ),
    )

    return Section(
        id="quality",
        title="Quality (filter ↔ annotate)",
        summary="MOS / SIGMOS / bandwidth gates. Annotate-only adds the metric without dropping rows.",
        questions=[mos_q, sigmos_q, band_q],
    )


def _annotations_section(intent: IntentCategories) -> Section:
    transcript_q = Question(
        id="text.transcript_source",
        intent_path="text.transcript_source",
        title="Transcripts?",
        why="Pulls in ASR or word-level alignment. Off means no text-related stages.",
        options=[
            Option(id="off", label="Off", value="off"),
            Option(id="generate", label="Generate via ASR", value="generate"),
            Option(id="existing", label="Use existing manifest column", value="existing"),
        ],
        prefill=Prefill(
            value=intent.text.transcript_source,
            source="prompt" if intent.text.transcript_source != "off" else "default",
            reason="Transcript source.",
        ),
    )

    word_timing_q = Question(
        id="text.word_timing",
        intent_path="text.word_timing",
        title="Word-level timestamps?",
        why="Switches generation to alignment, so every word has start/end times.",
        options=[
            Option(id="no", label="No", value=False),
            Option(id="yes", label="Yes", value=True),
        ],
        prefill=Prefill(value=intent.text.word_timing, source="default", reason="Defaults to off."),
        visible=intent.text.transcript_source == "generate",
        visible_when="text.transcript_source == generate",
    )

    asr_backend_q = Question(
        id="text.asr_backend",
        intent_path="text.asr_backend",
        title="ASR backend?",
        why="Auto picks NeMo by default; whisper switches to the WhisperX pair.",
        options=[
            Option(id="auto", label="Auto", value="auto"),
            Option(id="nemo", label="NeMo", value="nemo"),
            Option(id="whisper", label="Whisper", value="whisper"),
        ],
        prefill=Prefill(value=intent.text.asr_backend, source="default", reason="Defaults to auto (NeMo)."),
        visible=intent.text.transcript_source == "generate",
        visible_when="text.transcript_source == generate",
    )

    wer_q = Question(
        id="text.wer_mode",
        intent_path="text.wer_mode",
        title="WER gate (vs. existing transcript)?",
        why="Run WER between generated and reference text; annotate or drop if over threshold.",
        options=[
            Option(id="off", label="Off", value=FilterMode.OFF.value),
            Option(id="annotate", label="Annotate WER", value=FilterMode.ANNOTATE.value),
            Option(
                id="filter",
                label="Filter — drop WER > 0.5",
                value=FilterMode.FILTER.value,
                apply={"text.wer_max": 0.5},
            ),
            Option(
                id="custom",
                label="Filter — write your own WER ceiling",
                is_freeform=True,
                freeform_kind="float",
                freeform_placeholder="e.g. 0.35",
                apply={"text.wer_mode": FilterMode.FILTER.value},
            ),
        ],
        prefill=Prefill(value=intent.text.wer_mode.value, source="default", reason="Default off."),
        visible=intent.text.transcript_source != "off",
        visible_when="text.transcript_source != off",
    )

    speakers_q = Question(
        id="speakers.mode",
        intent_path="speakers.mode",
        title="Speaker handling?",
        why="Annotate adds diarization labels; filter restricts the row count; split fans out per-speaker.",
        options=[
            Option(id="off", label="Off", value=FilterMode.OFF.value),
            Option(id="annotate", label="Annotate — diarize and tag manifest", value=FilterMode.ANNOTATE.value),
            Option(
                id="filter_exact",
                label="Filter — exactly N speakers",
                value=FilterMode.FILTER.value,
                apply={"speakers.target_count": 1, "speakers.max_count": None},
            ),
            Option(
                id="filter_at_most",
                label="Filter — at most N speakers",
                value=FilterMode.FILTER.value,
                apply={"speakers.target_count": None, "speakers.max_count": 2},
            ),
            Option(id="split", label="Split — one row per speaker", value=FilterMode.SPLIT.value),
        ],
        prefill=Prefill(
            value=intent.speakers.mode.value,
            source="prompt" if intent.speakers.mode != FilterMode.OFF else "default",
            reason="Speaker mode.",
        ),
    )

    speaker_count_q = Question(
        id="speakers.count",
        intent_path="speakers.target_count",
        title="How many speakers?",
        why="Drives the value-filter on num_speakers when mode is filter.",
        options=[
            Option(id="1", label="1", value=None, apply={"speakers.target_count": 1, "speakers.max_count": None}),
            Option(id="2", label="2", value=None, apply={"speakers.target_count": 2, "speakers.max_count": None}),
            Option(id="3", label="3", value=None, apply={"speakers.target_count": 3, "speakers.max_count": None}),
            Option(
                id="custom",
                label="Write your own N",
                is_freeform=True,
                freeform_kind="speaker_count",
                freeform_placeholder="e.g. 4",
            ),
        ],
        prefill=Prefill(value=intent.speakers.target_count, source="default", reason="Defaults to 1."),
        visible=intent.speakers.mode == FilterMode.FILTER,
        visible_when="speakers.mode == FILTER",
    )

    return Section(
        id="annotations",
        title="Annotations",
        summary="Transcripts, word timing, WER, speakers.",
        questions=[transcript_q, word_timing_q, asr_backend_q, wer_q, speakers_q, speaker_count_q],
    )


def _policy_section(intent: IntentCategories) -> Section:
    commercial_q = Question(
        id="policy.commercial_only",
        intent_path="policy.commercial_only",
        title="Commercial use?",
        why="If yes, the license gate blocks non-commercial / GPL / unknown-license stages and models.",
        options=[
            Option(id="yes", label="Yes — commercial use", value=True),
            Option(id="no", label="No — research OK", value=False),
        ],
        prefill=Prefill(
            value=intent.policy.commercial_only,
            source="default",
            reason="Defaults to no — flip on for commercial pipelines.",
        ),
    )

    privacy_q = Question(
        id="policy.privacy_mode",
        intent_path="policy.privacy_mode",
        title="Privacy mode?",
        why="Future use: blocks remote model downloads / telemetry once the runner enforces it.",
        options=[
            Option(id="yes", label="Yes", value=True),
            Option(id="no", label="No", value=False),
        ],
        prefill=Prefill(
            value=intent.policy.privacy_mode,
            source="default",
            reason="Defaults to off.",
        ),
    )

    return Section(
        id="policy",
        title="Policy",
        summary="Licensing and privacy posture.",
        questions=[commercial_q, privacy_q],
    )


# ----------------------------------------------------------------------------
# Apply answers
# ----------------------------------------------------------------------------


def apply_answers(intent: IntentCategories, answers: dict[str, Any]) -> IntentCategories:
    """Merge a flat ``{intent_path: value}`` dict into ``intent``.

    ``answers`` may also carry option-level apply maps under the
    convention ``__apply__: {dotted.path: value, ...}`` — these are
    written first, then explicit per-path entries (so explicit entries
    win).
    """

    data = intent.model_dump(mode="json")
    notes = list(data.get("notes") or [])

    side_effects = answers.get("__apply__") or {}
    if isinstance(side_effects, dict):
        for path, value in side_effects.items():
            _set_path(data, str(path), value)

    for path, value in answers.items():
        if path == "__apply__":
            continue
        _set_path(data, str(path), value)
        notes.append(f"clarifier_answer:{path}={value}")

    data["notes"] = _dedupe(notes)
    return IntentCategories(**data)


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            return
        cursor.setdefault(part, {})
        cursor = cursor[part]
    if isinstance(cursor, dict):
        cursor[parts[-1]] = value


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def _parse_sample_rate(text: str) -> int | None:
    m = _SR_PATTERN.search(text)
    if m:
        khz = float(m.group(1))
        return int(round(khz * 1000))
    m = _SR_HZ_PATTERN.search(text)
    if m:
        return int(m.group(1))
    return None


def _parse_duration_range(text: str) -> tuple[float, float] | None:
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*(s|sec|second|seconds|min|minute|minutes)\b", text)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2))
    unit = m.group(3)
    if unit.startswith("min"):
        lo *= 60.0
        hi *= 60.0
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _duration_prefill(intent: IntentCategories) -> Prefill | None:
    seg = intent.segmentation
    if seg.duration_min_sec is None and seg.duration_max_sec is None:
        return None
    return Prefill(
        value=[seg.duration_min_sec, seg.duration_max_sec],
        source="prompt" if seg.output_unit != "original_files" else "default",
        reason="Inferred duration range.",
    )


def _source_for_sample_rate(intent: IntentCategories, profile: DatasetCard | None) -> PrefillSource:
    if intent.output.sample_rate is None:
        return "default"
    if profile and profile.profile and profile.profile.sample_rates_hz:
        keys = list(profile.profile.sample_rates_hz.keys())
        if len(keys) == 1 and str(intent.output.sample_rate) == keys[0]:
            return "profile"
    return "prompt"


def _explain_sample_rate(intent: IntentCategories, profile: DatasetCard | None) -> str:
    if intent.output.sample_rate == "any":
        return "Set to 'any' — no resample stage will be inserted."
    if profile and profile.profile and profile.profile.sample_rates_hz:
        return "Inferred from dataset profile or prompt."
    return "Inferred from prompt."


def _source_for_channels(intent: IntentCategories, profile: DatasetCard | None) -> PrefillSource:
    if intent.output.channels is None:
        return "default"
    if profile and profile.profile and profile.profile.channel_distribution:
        keys = list(profile.profile.channel_distribution.keys())
        single = len(keys) == 1
        if single and (
            (keys[0] == "1" and intent.output.channels == "mono")
            or (keys[0] == "2" and intent.output.channels == "stereo")
        ):
            return "profile"
    return "prompt"


def _explain_channels(intent: IntentCategories, profile: DatasetCard | None) -> str:
    if profile and profile.profile and profile.profile.channel_distribution:
        return "Inferred from dataset channel distribution."
    return "Inferred from prompt."


def _should_show_channels(profile: DatasetCard | None, intent: IntentCategories) -> bool:
    if intent.output.channels is not None:
        return True
    if profile is None or not profile.profile.channel_distribution:
        return True
    return len(profile.profile.channel_distribution) > 1


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


__all__ = [
    "ClarificationForm",
    "Option",
    "Prefill",
    "Question",
    "Section",
    "apply_answers",
    "apply_profile_prefills",
    "build_clarification_form",
    "infer_intent_from_prompt",
]
