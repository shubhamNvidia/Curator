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
"""Tests for :mod:`nemo_curator.agentic.intent`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_curator.agentic.cards import CapabilityTag
from nemo_curator.agentic.intent import IntentCategories, required_capabilities


class TestIntentCategoriesDefaults:
    def test_all_defaults_unset(self) -> None:
        ic = IntentCategories()
        assert ic.sample_rate is None
        assert ic.duration_min_sec is None
        assert ic.duration_max_sec is None
        assert ic.output_extract_clips is True  # the one non-None default

    def test_duration_order(self) -> None:
        with pytest.raises(ValidationError):
            IntentCategories(duration_min_sec=10.0, duration_max_sec=5.0)

    def test_known_sample_rate(self) -> None:
        IntentCategories(sample_rate=16000)
        IntentCategories(sample_rate="any")
        with pytest.raises(ValidationError):
            IntentCategories(sample_rate=14000)

    def test_unknown_keys_are_silently_dropped(self) -> None:
        """The schema is extra='ignore' so LLM-invented fields don't crash planning."""
        ic = IntentCategories.model_validate({"language": "en", "budgets": {"gpu_hours": 4}})
        # Unknown keys must NOT become attributes on the model.
        assert not hasattr(ic, "language")
        assert not hasattr(ic, "budgets")


class TestRequiredCapabilities:
    def test_only_clip_extraction_keeps_vad_plus_extract(self) -> None:
        ic = IntentCategories()
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.VAD in caps
        assert CapabilityTag.SEGMENT_EXTRACT in caps

    def test_mos_floor_triggers_mos_filter(self) -> None:
        ic = IntentCategories(quality_mos_min=3.5)
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.QUALITY_FILTER_MOS in caps

    def test_speakers_single_routes_to_separation(self) -> None:
        """speakers=1 → SpeakerSeparationStage (safe default for single-speaker output)."""
        ic = IntentCategories(speakers=1)
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.SPEAKER_SEPARATION in caps
        assert CapabilityTag.SPEAKER_DIARIZATION not in caps

    def test_speakers_multi_routes_to_diarization(self) -> None:
        """speakers>1 → diarization + value-filter on num_speakers (no separation)."""
        ic = IntentCategories(speakers=3)
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.SPEAKER_DIARIZATION in caps
        assert CapabilityTag.SPEAKER_SEPARATION not in caps

    def test_speakers_any_does_not_require_speaker_capabilities(self) -> None:
        """'any' speakers means the agent has no speaker-side constraint."""
        ic = IntentCategories(speakers="any")
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.SPEAKER_DIARIZATION not in caps
        assert CapabilityTag.SPEAKER_SEPARATION not in caps

    def test_wer_max_triggers_asr_and_wer(self) -> None:
        ic = IntentCategories(asr_wer_max=25.0)
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.ASR in caps
        assert CapabilityTag.WER in caps

    def test_alm_window_triggers_alm_package(self) -> None:
        ic = IntentCategories(alm_window_sec=120.0, alm_overlap_dedupe=True)
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.ALM_PACKAGE in caps
        assert CapabilityTag.ALM_OVERLAP in caps

    def test_unsupported_intents_are_dropped_not_persisted(self) -> None:
        """Phase 1 schema doesn't know about language/emotion/etc. — they must be silently
        dropped (with extra='ignore'), not become accidental hidden state."""
        for missing in ("language", "emotion", "accent", "gender", "quality_dnsmos_min", "quality_snr_min_db"):
            ic = IntentCategories.model_validate({missing: "en"})
            assert not hasattr(ic, missing)
            assert missing not in ic.model_dump()
