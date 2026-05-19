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
"""Schema-level tests for ``nemo_curator.agentic.cards``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_curator.agentic.cards import (
    Cardinality,
    CapabilityTag,
    DatasetCard,
    IOSpec,
    LicenseKind,
    ModelRef,
    ParamSpec,
    ResourceSpec,
    RunCard,
    SelectionHints,
    StageCard,
    StageCategory,
    ThresholdBand,
    ThresholdCombo,
    ThresholdSetting,
)


def _minimal_stage_card(**overrides) -> StageCard:
    base = dict(
        name="VADSegmentationStage",
        target="nemo_curator.stages.audio.segmentation.vad_segmentation.VADSegmentationStage",
        description="VAD segmentation.",
        summary="Splits audio into VAD-bounded segments.",
        category=StageCategory.SEGMENTATION,
        capabilities=[CapabilityTag.VAD],
        produces_cardinality=Cardinality.ONE_TO_MANY,
    )
    base.update(overrides)
    return StageCard(**base)


class TestStageCardBasics:
    def test_minimal_round_trips(self) -> None:
        card = _minimal_stage_card()
        assert card.name == "VADSegmentationStage"
        assert CapabilityTag.VAD in card.capabilities
        assert card.produces_cardinality == Cardinality.ONE_TO_MANY
        # Dumping then re-loading should be lossless.
        again = StageCard.model_validate(card.model_dump(mode="json"))
        assert again == card

    def test_name_must_be_identifier(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_stage_card(name="not a class")
        with pytest.raises(ValidationError):
            _minimal_stage_card(name="9StartsWithDigit")

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            StageCard.model_validate({
                **_minimal_stage_card().model_dump(mode="json"),
                "totally_made_up": True,
            })


class TestResourceSpec:
    def test_default(self) -> None:
        r = ResourceSpec()
        assert r.cpus == 1.0
        assert r.gpus == 0.0
        assert r.gpu_memory_gb == 0.0

    def test_gpus_xor_gpu_memory_gb(self) -> None:
        with pytest.raises(ValidationError):
            ResourceSpec(gpus=1.0, gpu_memory_gb=8.0)

    def test_to_resources_kwargs_strips_zeros(self) -> None:
        kwargs = ResourceSpec(cpus=2.0).to_resources_kwargs()
        assert kwargs == {"cpus": 2.0}
        kwargs = ResourceSpec(cpus=1.0, gpus=0.5).to_resources_kwargs()
        assert kwargs == {"cpus": 1.0, "gpus": 0.5}
        kwargs = ResourceSpec(cpus=1.0, gpu_memory_gb=8.0).to_resources_kwargs()
        assert kwargs == {"cpus": 1.0, "gpu_memory_gb": 8.0}


class TestNestedCardinality:
    def test_nested_requires_segment_key(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_stage_card(produces_cardinality=Cardinality.ONE_TO_ONE_NESTED)

    def test_nested_with_segment_key(self) -> None:
        card = _minimal_stage_card(
            produces_cardinality=Cardinality.ONE_TO_ONE_NESTED,
            nested_segment_key="segments",
        )
        assert card.nested_segment_key == "segments"


class TestIOSpec:
    def test_default_empty(self) -> None:
        io = IOSpec()
        assert io.as_tuple() == ([], [])

    def test_round_trip(self) -> None:
        io = IOSpec(top_level=["filepath_key"], data=["audio_filepath"])
        assert io.as_tuple() == (["filepath_key"], ["audio_filepath"])


class TestLicenseConsistency:
    def test_commercial_safe_model_must_be_ok(self) -> None:
        card = _minimal_stage_card(
            license=LicenseKind.APACHE_2_0,
            commercial_safe=True,
            models=[ModelRef(name="foo", provider="huggingface", license=LicenseKind.MIT)],
        )
        assert card.commercial_safe is True

    def test_card_apache_but_nc_model_blocked_under_commercial(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_stage_card(
                license=LicenseKind.CC_BY_NC_4_0,
                commercial_safe=True,
                models=[ModelRef(name="bad", provider="huggingface", license=LicenseKind.GPL_3_0)],
            )


class TestSelectionHints:
    def test_threshold_bands_round_trip(self) -> None:
        """A stage card can carry per-param threshold guidance for the planner."""
        hints = SelectionHints(
            prefer_when=["clean output requested"],
            threshold_bands=[
                ThresholdBand(param="mos_threshold", value=3.5, label="clean"),
                ThresholdBand(param="mos_threshold", value=4.0, label="studio"),
            ],
        )
        dumped = hints.model_dump(mode="json")
        assert dumped["threshold_bands"][0]["param"] == "mos_threshold"
        assert dumped["threshold_bands"][1]["value"] == 4.0
        # Round-trip is lossless.
        SelectionHints.model_validate(dumped)

    def test_threshold_band_rejects_extra_fields(self) -> None:
        """Extra fields on ThresholdBand must be rejected so cards stay tight."""
        with pytest.raises(ValidationError):
            ThresholdBand.model_validate({
                "param": "mos_threshold", "value": 3.5, "label": "clean",
                "extra_field": "boom",
            })

    def test_combo_preset_round_trip(self) -> None:
        """Cross-stage combo presets must validate and survive a JSON round-trip."""
        hints = SelectionHints(
            combo_presets=[
                ThresholdCombo(
                    label="clean speech (TTS default)",
                    aliases=["clean", "TTS-ready"],
                    description="Project default for TTS curation.",
                    settings=[
                        ThresholdSetting(stage="UTMOSFilterStage", param="mos_threshold", value=3.4),
                        ThresholdSetting(stage="SIGMOSFilterStage", param="noise_threshold", value=4.0),
                        ThresholdSetting(stage="SIGMOSFilterStage", param="ovrl_threshold", value=3.5),
                    ],
                )
            ]
        )
        dumped = hints.model_dump(mode="json")
        combo = dumped["combo_presets"][0]
        assert combo["label"] == "clean speech (TTS default)"
        assert {s["stage"] for s in combo["settings"]} == {"UTMOSFilterStage", "SIGMOSFilterStage"}
        SelectionHints.model_validate(dumped)

    def test_threshold_setting_accepts_string_value(self) -> None:
        """BandFilterStage uses an enum (full_band / narrow_band) param — combos must
        carry string values, not just floats."""
        s = ThresholdSetting(stage="BandFilterStage", param="band_value", value="full_band")
        assert s.value == "full_band"
        # Float values still work for MOS-style params.
        f = ThresholdSetting(stage="UTMOSFilterStage", param="mos_threshold", value=3.5)
        assert f.value == 3.5
        # A combo can mix both.
        combo = ThresholdCombo(label="clean", settings=[s, f])
        dumped = combo.model_dump(mode="json")
        ThresholdCombo.model_validate(dumped)


class TestSiblingCards:
    def test_dataset_card_minimum(self) -> None:
        dc = DatasetCard(name="ds", uri="file:///tmp/x.jsonl")
        assert dc.uri_scheme == "file"

    def test_run_card_minimum(self) -> None:
        rc = RunCard(
            run_id="r1",
            started_at="2026-05-12T10:00:00",
            target_dir="/tmp/out",
        )
        assert rc.success is False
        assert rc.executor == "xenna"
