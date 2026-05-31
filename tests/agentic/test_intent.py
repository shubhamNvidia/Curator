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
"""Tests for :mod:`nemo_curator.agentic.intent` (V2 ingredient schema)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_curator.agentic.cards import CapabilityTag
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    Policy,
    Quality,
    Segmentation,
    Speakers,
    TextPolicy,
    required_capabilities,
)


class TestIntentCategoriesDefaults:
    def test_all_submodels_default_unset(self) -> None:
        ic = IntentCategories()
        assert ic.output.sample_rate is None
        assert ic.output.channels is None
        assert ic.output.audio_format == "wav"
        assert ic.segmentation.output_unit == "original_files"
        assert ic.segmentation.speech_policy == FilterMode.OFF
        assert ic.quality.mos == FilterMode.OFF
        assert ic.quality.sigmos == FilterMode.OFF
        assert ic.quality.band == FilterMode.OFF
        assert ic.speakers.mode == FilterMode.OFF
        assert ic.text.transcript_source == "off"
        assert ic.text.wer_mode == FilterMode.OFF
        assert ic.policy.commercial_only is False
        assert ic.policy.privacy_mode is False

    def test_duration_min_max_order(self) -> None:
        with pytest.raises(ValidationError):
            IntentCategories(segmentation=Segmentation(
                duration_min_sec=10.0, duration_max_sec=5.0,
            ))

    def test_speakers_filter_without_count_records_note(self) -> None:
        ic = IntentCategories(speakers=Speakers(mode=FilterMode.FILTER))
        assert any("speakers.mode=FILTER" in n for n in ic.notes)

    def test_long_windows_without_window_sec_records_note(self) -> None:
        ic = IntentCategories(segmentation=Segmentation(output_unit="long_windows"))
        assert any("long_windows" in n for n in ic.notes)

    def test_speech_segments_with_speakers_split_coerces_to_single_speaker_clips(self) -> None:
        """``speakers.mode=SPLIT`` + ``output_unit=speech_segments`` is
        the same intent as ``single_speaker_clips``. The validator
        normalizes it so the selector produces the canonical
        SpeakerSep→VAD→quality order instead of the broken
        VAD→Concat→SpeakerSep→PBV order.

        Regression for run 06578e4b178f11e3 where VAD's duration
        window was applied at the file level, then concat threw the
        windows away, then SpeakerSep produced full-length per-speaker
        stems that no longer respected the user's 1-10 s cap.
        """

        ic = IntentCategories(
            segmentation=Segmentation(
                output_unit="speech_segments",
                duration_min_sec=1.0,
                duration_max_sec=10.0,
            ),
            speakers=Speakers(mode=FilterMode.SPLIT),
        )
        assert ic.segmentation.output_unit == "single_speaker_clips"
        assert any(
            "coerced from 'speech_segments' to 'single_speaker_clips'" in n
            for n in ic.notes
        )
        # Duration window is preserved through the coercion.
        assert ic.segmentation.duration_min_sec == 1.0
        assert ic.segmentation.duration_max_sec == 10.0

    def test_speech_segments_without_split_is_not_coerced(self) -> None:
        """Coercion is gated on speakers.mode=SPLIT. Other modes leave
        ``speech_segments`` alone (it's a valid output unit on its own)."""

        for mode in (FilterMode.OFF, FilterMode.ANNOTATE, FilterMode.FILTER):
            ic = IntentCategories(
                segmentation=Segmentation(output_unit="speech_segments"),
                speakers=Speakers(mode=mode),
            )
            assert ic.segmentation.output_unit == "speech_segments", mode

    def test_extra_keys_silently_dropped_top_level(self) -> None:
        """V2 still uses extra='ignore' so unknown LLM keys don't crash."""
        ic = IntentCategories.model_validate({"language": "en", "budgets": {"gpu_hours": 4}})
        assert not hasattr(ic, "language")
        assert not hasattr(ic, "budgets")

    def test_v1_unnamespaced_keys_silently_dropped_to_v2_defaults(self) -> None:
        """V1-flat keys that are no longer top-level fields are dropped to defaults.

        ``speakers`` is intentionally NOT in this list because V2 still has
        a top-level ``speakers`` key (now a submodel), so a V1 integer
        ``speakers=1`` would now correctly raise a typed validation error
        — surfacing the schema break to the LLM rather than silently
        coercing.
        """
        ic = IntentCategories.model_validate({
            "sample_rate": 48000,
            "channels": "mono",
            "quality_mos_min": 3.5,
            "duration_min_sec": 2.0,
            "duration_max_sec": 60.0,
            "output_extract_clips": True,
        })
        assert ic.output.sample_rate is None
        assert ic.output.channels is None
        assert ic.quality.mos == FilterMode.OFF
        assert ic.segmentation.duration_min_sec is None

    def test_namespaced_construction_works(self) -> None:
        ic = IntentCategories(
            output={"sample_rate": 48000, "channels": "mono"},
            quality={"mos": FilterMode.FILTER, "mos_threshold": 3.5},
            speakers={"mode": FilterMode.SPLIT},
        )
        assert ic.output.sample_rate == 48000
        assert ic.output.channels == "mono"
        assert ic.quality.mos == FilterMode.FILTER
        assert ic.quality.mos_threshold == 3.5
        assert ic.speakers.mode == FilterMode.SPLIT


class TestRequiredCapabilities:
    def test_default_intent_has_no_capabilities(self) -> None:
        ic = IntentCategories()
        assert required_capabilities(ic) == []

    def test_speech_segments_requires_vad_and_extract(self) -> None:
        ic = IntentCategories(segmentation=Segmentation(output_unit="speech_segments"))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.VAD in caps
        assert CapabilityTag.SEGMENT_EXTRACT in caps

    def test_single_speaker_clips_pulls_separation_and_extract(self) -> None:
        """``single_speaker_clips`` is segmented by ``SpeakerSeparationStage``;
        VAD is intentionally NOT required (forcing VAD upfront would
        double-segment the audio and break the one-clip-per-speaker
        semantic). The cleaning flow is reserved for ``original_files``."""

        ic = IntentCategories(segmentation=Segmentation(output_unit="single_speaker_clips"))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.VAD not in caps
        assert CapabilityTag.SPEAKER_SEPARATION in caps
        assert CapabilityTag.SEGMENT_EXTRACT in caps

    def test_speech_policy_filter_pulls_vad_even_on_original_files(self) -> None:
        ic = IntentCategories(segmentation=Segmentation(
            output_unit="original_files",
            speech_policy=FilterMode.FILTER,
        ))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.VAD in caps

    def test_mos_annotate_still_requires_quality_filter_capability(self) -> None:
        """ANNOTATE = same stage as FILTER, threshold-zero — still needs the cap."""
        ic = IntentCategories(quality=Quality(mos=FilterMode.ANNOTATE))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.QUALITY_FILTER_MOS in caps

    def test_mos_filter_threshold_pulls_quality_cap(self) -> None:
        ic = IntentCategories(quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.QUALITY_FILTER_MOS in caps

    def test_band_filter_requires_band_capability(self) -> None:
        ic = IntentCategories(quality=Quality(band=FilterMode.FILTER, band_value="narrow_band"))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.QUALITY_FILTER_BAND in caps

    def test_speakers_split_routes_to_separation(self) -> None:
        ic = IntentCategories(speakers=Speakers(mode=FilterMode.SPLIT))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.SPEAKER_SEPARATION in caps
        assert CapabilityTag.SPEAKER_DIARIZATION not in caps

    def test_speakers_annotate_routes_to_diarization(self) -> None:
        ic = IntentCategories(speakers=Speakers(mode=FilterMode.ANNOTATE))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.SPEAKER_DIARIZATION in caps
        assert CapabilityTag.SPEAKER_SEPARATION not in caps

    def test_speakers_off_pulls_no_speaker_caps(self) -> None:
        ic = IntentCategories(speakers=Speakers(mode=FilterMode.OFF))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.SPEAKER_DIARIZATION not in caps
        assert CapabilityTag.SPEAKER_SEPARATION not in caps

    def test_transcript_generate_pulls_asr(self) -> None:
        ic = IntentCategories(text=TextPolicy(transcript_source="generate"))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.ASR in caps

    def test_word_timing_pulls_asr_align(self) -> None:
        ic = IntentCategories(text=TextPolicy(transcript_source="generate", word_timing=True))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.ASR_ALIGN in caps

    def test_wer_filter_pulls_wer_capability(self) -> None:
        ic = IntentCategories(text=TextPolicy(wer_mode=FilterMode.FILTER, wer_max=0.5))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.WER in caps

    def test_long_windows_pulls_alm_package(self) -> None:
        ic = IntentCategories(segmentation=Segmentation(
            output_unit="long_windows", long_window_sec=120.0,
        ))
        caps = {r.capability for r in required_capabilities(ic)}
        assert CapabilityTag.ALM_PACKAGE in caps
