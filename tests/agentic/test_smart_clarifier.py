# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the adaptive LLM-driven smart clarifier.

These tests exercise three orthogonal concerns:

1. ``analyze_gaps`` (pure deterministic policy) — the right fields are
   surfaced for the right prompts, and irrelevant families stay quiet.
2. The template fallback (no LLM available) — every gap maps to a
   well-formed :class:`SmartQuestion` with valid options.
3. The LLM composer using a :class:`MockLLM` — phrasing is taken from
   the LLM, but ids / intent paths / option values / ``apply`` /
   ``reveals`` are preserved verbatim from the template.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    OutputFormat,
    Quality,
    Segmentation,
    Speakers,
    TextPolicy,
)
from nemo_curator.agentic.llm import Message, MockLLM
from nemo_curator.agentic.smart_clarifier import (
    Gap,
    SmartForm,
    SmartOption,
    SmartQuestion,
    analyze_gaps,
    build_inferred_chips,
    compose_smart_form,
    template_questions,
)


# ----------------------------------------------------------------------------
# analyze_gaps
# ----------------------------------------------------------------------------


def test_analyze_gaps_essentials_when_empty_intent():
    """A blank intent surfaces sample_rate as a gap and stays quiet on
    families the user didn't mention."""

    gaps = analyze_gaps(IntentCategories(), prompt="just process my audio", profile=None)
    paths = [g.intent_path for g in gaps]
    assert "output.sample_rate" in paths
    # audio_format has a sensible "wav" default → no need to ask.
    assert "output.audio_format" not in paths
    # No quality / speaker / transcript wording → those gaps must NOT fire.
    assert "quality.mos" not in paths
    assert "speakers.mode" not in paths
    assert "text.transcript_source" not in paths


def test_analyze_gaps_audio_format_fires_only_when_cleared():
    """audio_format defaults to ``wav``; ask only when explicitly cleared."""

    cleared = IntentCategories(output=OutputFormat(sample_rate=16000, audio_format=None))
    default = IntentCategories(output=OutputFormat(sample_rate=16000))
    paths_cleared = [g.intent_path for g in analyze_gaps(cleared, "n/a", None)]
    paths_default = [g.intent_path for g in analyze_gaps(default, "n/a", None)]
    assert "output.audio_format" in paths_cleared
    assert "output.audio_format" not in paths_default


def test_analyze_gaps_suppresses_essentials_already_set():
    """Don't re-ask for sample_rate / format the extractor already filled."""

    intent = IntentCategories(
        output=OutputFormat(sample_rate=24000, audio_format="wav", resample_input=True)
    )
    gaps = analyze_gaps(intent, prompt="tts dataset", profile=None)
    paths = [g.intent_path for g in gaps]
    assert "output.sample_rate" not in paths
    assert "output.audio_format" not in paths
    assert "output.resample_input" not in paths


def test_analyze_gaps_duration_only_when_mentioned():
    """The duration follow-up only surfaces when the prompt hints at it."""

    intent = IntentCategories()
    no_duration = analyze_gaps(intent, "make a tts dataset", profile=None)
    yes_duration = analyze_gaps(intent, "make tts data with clips between 2 and 30 seconds", profile=None)
    assert "__duration_constraint__" not in [g.intent_path for g in no_duration]
    duration_gap = next(
        (g for g in yes_duration if g.intent_path == "__duration_constraint__"),
        None,
    )
    assert duration_gap is not None
    # It carries min + max as follow-ups, each with sensible suggested defaults.
    follow_paths = [fu.intent_path for fu in duration_gap.follow_ups]
    assert follow_paths == ["segmentation.duration_min_sec", "segmentation.duration_max_sec"]


def test_analyze_gaps_quality_only_when_mentioned():
    """Don't ask about MOS / SIGMOS unless the user used quality words."""

    intent = IntentCategories()
    silent = analyze_gaps(intent, "build an asr dataset", profile=None)
    spoken = analyze_gaps(intent, "build an asr dataset, drop noisy clips", profile=None)
    assert "quality.mos" not in [g.intent_path for g in silent]
    assert "quality.mos" in [g.intent_path for g in spoken]


def test_analyze_gaps_quality_fires_even_when_heuristic_prefilled():
    """When the prompt has quality wording we ALWAYS surface the
    aggressiveness question — even if the prompt heuristic already
    inferred a threshold — and we suggest the level matching that
    pre-fill so the user can keep or override it."""

    # Pre-filled by the heuristic for a "balanced" / "clean" prompt:
    # UTMOS ≥ 3.4 + SIGMOS ovrl/noise.
    intent = IntentCategories(
        quality=Quality(
            mos=FilterMode.FILTER,
            mos_threshold=3.4,
            sigmos=FilterMode.FILTER,
        ),
    )
    gaps = analyze_gaps(intent, "please clean my audio dataset", profile=None)
    quality_gap = next((g for g in gaps if g.intent_path == "quality.mos"), None)
    assert quality_gap is not None, (
        "Quality gap must still fire when the prompt has quality wording, "
        "even if the heuristic prefilled — otherwise the user can't "
        "override the inferred aggressiveness level."
    )
    assert quality_gap.suggested_value == "balanced", (
        f"Expected suggested aggressiveness='balanced' for MOS≥3.4 prefill; "
        f"got {quality_gap.suggested_value}"
    )


def test_quality_options_carry_utmos_sigmos_combinations():
    """Each aggressiveness level must apply UTMOS *and* SIGMOS together
    via the option's ``apply`` dict (with the sole exception of the
    ``light`` level, which turns SIGMOS off by design)."""

    intent = IntentCategories()
    gaps = analyze_gaps(intent, "clean my audio data, drop noisy", profile=None)
    qs = template_questions(gaps, intent)
    qual = next((q for q in qs if q.id == "q_quality"), None)
    assert qual is not None
    by_id = {o.id: o for o in qual.options}
    # Every aggressiveness level is present and well-formed.
    assert {"light", "balanced", "strict", "annotate", "off"} <= set(by_id.keys())

    balanced = by_id["balanced"]
    assert balanced.apply is not None
    assert balanced.apply["quality.mos"] == "filter"
    assert balanced.apply["quality.mos_threshold"] == 3.4
    assert balanced.apply["quality.sigmos"] == "filter"
    assert balanced.apply["quality.sigmos_axes"] == ["ovrl", "noise"]

    strict = by_id["strict"]
    assert strict.apply is not None
    assert strict.apply["quality.mos_threshold"] == 4.0
    assert strict.apply["quality.sigmos_thresholds"] == {"ovrl": 4.0, "noise": 4.0}

    annotate = by_id["annotate"]
    assert annotate.apply is not None
    assert annotate.apply["quality.mos"] == "annotate"
    assert annotate.apply["quality.sigmos"] == "annotate"


def test_analyze_gaps_speakers_only_when_mentioned():
    intent = IntentCategories()
    no = analyze_gaps(intent, "clean tts dataset", profile=None)
    yes = analyze_gaps(intent, "build a diarized meeting dataset, one speaker per row", profile=None)
    assert "speakers.mode" not in [g.intent_path for g in no]
    assert "speakers.mode" in [g.intent_path for g in yes]


def test_analyze_gaps_transcripts_only_when_mentioned():
    intent = IntentCategories()
    no = analyze_gaps(intent, "clean dataset for tts", profile=None)
    yes = analyze_gaps(intent, "asr dataset with transcripts please", profile=None)
    assert "text.transcript_source" not in [g.intent_path for g in no]
    assert "text.transcript_source" in [g.intent_path for g in yes]


def test_analyze_gaps_resample_only_when_rate_known():
    """resample_input is only worth asking once a concrete rate is set."""

    intent_no_rate = IntentCategories()
    intent_any = IntentCategories(output=OutputFormat(sample_rate="any"))
    intent_concrete = IntentCategories(output=OutputFormat(sample_rate=16000))

    paths_no = [g.intent_path for g in analyze_gaps(intent_no_rate, "noop", None)]
    paths_any = [g.intent_path for g in analyze_gaps(intent_any, "noop", None)]
    paths_concrete = [g.intent_path for g in analyze_gaps(intent_concrete, "noop", None)]

    assert "output.resample_input" not in paths_no
    assert "output.resample_input" not in paths_any
    assert "output.resample_input" in paths_concrete


# ----------------------------------------------------------------------------
# Template fallback
# ----------------------------------------------------------------------------


def test_template_questions_well_formed_for_every_gap_category():
    """Every gap category emitted by analyze_gaps has a template handler."""

    intent = IntentCategories()
    prompt = (
        "Build an ASR dataset, drop noisy clips, between 2 and 30 second clips, "
        "speaker labels please, with transcripts."
    )
    gaps = analyze_gaps(intent, prompt, profile=None)
    qs = template_questions(gaps, intent)
    assert len(qs) >= 5, "expected multiple template questions for this rich prompt"

    # Every question must have a non-empty title, a unique id, at least
    # one option, and at least one option with a non-None ``value`` (so
    # the user can actually submit an answer).
    seen_ids: set[str] = set()
    for q in qs:
        assert q.title
        assert q.id not in seen_ids
        seen_ids.add(q.id)
        assert q.options, f"{q.id} has no options"
        # Picking each non-freeform option must produce a concrete answer.
        for o in q.options:
            assert o.id
            assert o.label
            if not o.is_freeform:
                # Either a direct value or an apply dict must exist.
                assert (o.value is not None) or (o.apply is not None), q.id


def test_template_duration_question_has_inline_followups():
    """The duration enable question carries min + max as follow_ups."""

    intent = IntentCategories()
    gaps = analyze_gaps(intent, "give me short clips", profile=None)
    qs = template_questions(gaps, intent)
    dur = next((q for q in qs if q.id == "q_duration_enable"), None)
    assert dur is not None
    assert {fu.id for fu in dur.follow_ups} == {"q_duration_min", "q_duration_max"}
    yes_option = next(o for o in dur.options if o.id == "yes")
    assert "q_duration_min" in yes_option.reveals
    assert "q_duration_max" in yes_option.reveals


# ----------------------------------------------------------------------------
# LLM composer (MockLLM)
# ----------------------------------------------------------------------------


def _responder_for(payload: dict[str, Any]):
    """Helper: build a MockLLM that returns ``payload`` as JSON regardless of input."""

    def respond(_messages: list[Message], _tier: str) -> str:
        return json.dumps(payload)

    return respond


def test_llm_composer_preserves_template_values_and_apply():
    """The LLM is only allowed to rewrite strings; ids / values / apply
    must survive verbatim from the template."""

    intent = IntentCategories()
    # Deliberately *neutral* prompt: the legacy regex heuristics in
    # ``infer_intent_from_prompt`` would otherwise pre-fill sample_rate
    # / mos and the corresponding gaps wouldn't fire.
    prompt = "process my audio files, drop noisy clips"

    # Fake LLM rewrites titles and labels but tries (incorrectly) to also
    # mutate option values. The composer must ignore the value rewrite.
    fake_llm = MockLLM(
        responder=_responder_for(
            {
                "questions": [
                    {
                        "id": "q_sample_rate",
                        "intent_path": "output.sample_rate",
                        "title": "Pick a sample rate",
                        "detail": "16 kHz works for ASR; 24 kHz for TTS.",
                        "options": [
                            {"id": "24k", "label": "24 kHz ✨", "value": "MALICIOUS"},
                            {"id": "16k", "label": "16 kHz (ASR-style)"},
                        ],
                    },
                    {
                        "id": "q_quality",
                        "intent_path": "quality.mos",
                        "title": "How aggressive on cleaning?",
                        "options": [
                            {"id": "balanced", "label": "Drop noisy"},
                            {"id": "off", "label": "Skip"},
                        ],
                    },
                ]
            }
        )
    )

    form = compose_smart_form(prompt, intent, profile=None, llm=fake_llm)
    assert isinstance(form, SmartForm)

    by_id = {q.id: q for q in form.questions}
    sr = by_id["q_sample_rate"]
    assert sr.title == "Pick a sample rate"           # LLM phrasing kept
    twentyfour = next(o for o in sr.options if o.id == "24k")
    assert twentyfour.label == "24 kHz ✨"             # label rewrite kept
    assert twentyfour.value == 24000                    # value preserved from template (not MALICIOUS)
    # The aggressiveness levels (balanced / strict / light / annotate /
    # off) survive the LLM round-trip with their multi-path apply dicts
    # intact — only the user-facing labels are rewritten.
    qual = by_id["q_quality"]
    balanced = next(o for o in qual.options if o.id == "balanced")
    assert balanced.label == "Drop noisy"                # label rewrite kept
    assert balanced.apply == {
        "quality.mos": "filter",
        "quality.mos_threshold": 3.4,
        "quality.sigmos": "filter",
        "quality.sigmos_axes": ["ovrl", "noise"],
        "quality.sigmos_thresholds": {"ovrl": 3.5, "noise": 4.0},
    }


def test_llm_composer_falls_back_to_template_on_bad_payload():
    """If the LLM returns garbage, we still get the deterministic form."""

    intent = IntentCategories()
    prompt = "make a dataset for me"   # vague enough that gap analyzer fires sample_rate
    fake_llm = MockLLM(responder=_responder_for({"not_questions": []}))

    form = compose_smart_form(prompt, intent, profile=None, llm=fake_llm)
    paths = [q.intent_path for q in form.questions]
    # Fallback path always surfaces the always-essential sample-rate gap.
    assert "output.sample_rate" in paths
    # And every question is fully formed.
    for q in form.questions:
        assert q.title and q.options


def test_llm_composer_preserves_reveals_on_duration_yes():
    """Even if the LLM only sends a subset of options, the reveals on the
    yes-option must still be there or the runtime would silently break
    conditional follow-ups."""

    intent = IntentCategories()
    # Plain English mentioning clips; no "between X and Y" range or
    # quality words that would otherwise trigger heuristic auto-fills
    # for sample_rate or speakers.
    prompt = "give me short clips please"
    fake_llm = MockLLM(
        responder=_responder_for(
            {
                "questions": [
                    {
                        "id": "q_duration_enable",
                        "intent_path": "__duration_constraint__",
                        "title": "Filter by length?",
                        "options": [
                            # LLM dropped 'yes' (mistake) — composer must
                            # re-attach it because it has a non-empty
                            # ``reveals`` list the runtime depends on.
                            {"id": "no", "label": "Any length"},
                        ],
                    }
                ]
            }
        )
    )
    form = compose_smart_form(prompt, intent, profile=None, llm=fake_llm)
    dur = next(q for q in form.questions if q.id == "q_duration_enable")
    option_ids = {o.id for o in dur.options}
    assert {"no", "yes"} <= option_ids
    yes_opt = next(o for o in dur.options if o.id == "yes")
    assert "q_duration_min" in yes_opt.reveals
    assert "q_duration_max" in yes_opt.reveals


# ----------------------------------------------------------------------------
# Inferred chips
# ----------------------------------------------------------------------------


def test_build_inferred_chips_surfaces_extractor_choices():
    """Anything the extractor settled shows up as a chip with a source label."""

    intent = IntentCategories(
        output=OutputFormat(sample_rate=24000, channels="mono", audio_format="wav"),
        quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.4),
    )
    chips = build_inferred_chips(intent, prompt="clean tts at 24 kHz mono", profile=None)
    paths = [c.intent_path for c in chips]
    assert "output.sample_rate" in paths
    assert "output.channels" in paths
    assert "output.audio_format" in paths
    assert "quality.mos" in paths
    # Every chip carries a source tag the UI can colour-code.
    for chip in chips:
        assert chip.source in {"prompt", "profile", "default"}


def test_compose_smart_form_advanced_form_always_present():
    """The advanced (legacy) form is always emitted as a back-pocket
    fallback, even when the smart form already covers everything."""

    intent = IntentCategories(
        output=OutputFormat(sample_rate=24000, audio_format="wav", resample_input=True),
    )
    form = compose_smart_form(
        "tts dataset",
        intent,
        profile=None,
        llm=None,
    )
    # No essentials missing → smart questions may be empty.
    assert form.advanced_form is not None
    section_ids = [s.id for s in form.advanced_form.sections]
    assert section_ids == ["output", "segmentation", "quality", "annotations", "policy"]
