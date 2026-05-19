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
"""Tests for the deterministic-only critic (Layer 5 stub)."""

from __future__ import annotations

from datetime import datetime, timezone

from nemo_curator.agentic.cards import (
    Cardinality,
    DatasetCard,
    DatasetProfile,
    RunCard,
    RunStageRecord,
)
from nemo_curator.agentic.critic import CriticOptions, critique
from nemo_curator.agentic.intent import IntentCategories


def _run_card(records: list[RunStageRecord]) -> RunCard:
    return RunCard(
        run_id="r",
        started_at=datetime.now(timezone.utc),
        target_dir="/tmp/out",
        stage_records=records,
        success=True,
    )


def _stage_rec(name: str, tasks_in: int, tasks_out: int) -> RunStageRecord:
    return RunStageRecord(
        stage_name=name,
        target=f"x.{name}",
        cardinality=Cardinality.ONE_TO_ONE,
        tasks_in=tasks_in,
        tasks_out=tasks_out,
        elapsed_sec=0.1,
    )


class TestDropRates:
    def test_no_drop_no_findings(self) -> None:
        rc = _run_card([_stage_rec("A", 10, 10)])
        rep = critique(run_card=rc)
        assert rep.drop_rates_by_stage["A"] == 0.0
        assert rep.deterministic_pass is True

    def test_high_drop_triggers_error_level_note(self) -> None:
        rc = _run_card([_stage_rec("UTMOSFilterStage", 100, 5)])  # 95% drop
        rep = critique(run_card=rc, options=CriticOptions(max_drop_rate_error=0.9))
        assert rep.deterministic_pass is False
        assert rep.drop_rates_by_stage["UTMOSFilterStage"] == 0.95
        joined = " ".join(rep.deterministic_findings)
        assert "UTMOSFilterStage" in joined

    def test_moderate_drop_triggers_warning(self) -> None:
        rc = _run_card([_stage_rec("VADSegmentationStage", 100, 40)])  # 60% drop
        rep = critique(run_card=rc, options=CriticOptions(max_drop_rate_warning=0.5, max_drop_rate_error=0.9))
        assert "VADSegmentationStage" in rep.drop_rates_by_stage
        assert rep.deterministic_pass is False  # warning still flips deterministic_pass


class TestDurationDrift:
    def test_large_p50_shift_emits_finding(self) -> None:
        in_card = DatasetCard(
            name="in",
            uri="x",
            profile=DatasetProfile(total_files=10, decodable_files=10, duration_p50_sec=4.0),
        )
        out_card = DatasetCard(
            name="out",
            uri="y",
            profile=DatasetProfile(total_files=10, decodable_files=10, duration_p50_sec=1.0),
        )
        rep = critique(run_card=_run_card([]), input_card=in_card, output_card=out_card)
        joined = " ".join(rep.deterministic_findings)
        assert "Median duration" in joined

    def test_small_shift_is_quiet(self) -> None:
        in_card = DatasetCard(
            name="in", uri="x",
            profile=DatasetProfile(total_files=10, decodable_files=10, duration_p50_sec=4.0),
        )
        out_card = DatasetCard(
            name="out", uri="y",
            profile=DatasetProfile(total_files=10, decodable_files=10, duration_p50_sec=4.1),
        )
        rep = critique(run_card=_run_card([]), input_card=in_card, output_card=out_card)
        assert rep.deterministic_pass is True


class TestIntentAlignment:
    def test_p95_above_intent_max(self) -> None:
        intent = IntentCategories(duration_max_sec=10.0)
        out_card = DatasetCard(
            name="out", uri="y",
            profile=DatasetProfile(total_files=10, decodable_files=10, duration_p95_sec=20.0),
        )
        rep = critique(run_card=_run_card([]), intent=intent, output_card=out_card)
        joined = " ".join(rep.deterministic_findings)
        assert "exceeds intent.duration_max_sec" in joined

    def test_sample_rate_mismatch(self) -> None:
        intent = IntentCategories(sample_rate=48000)
        out_card = DatasetCard(
            name="out", uri="y",
            profile=DatasetProfile(total_files=10, decodable_files=10, sample_rates_hz={"16000": 10}),
        )
        rep = critique(run_card=_run_card([]), intent=intent, output_card=out_card)
        joined = " ".join(rep.deterministic_findings)
        assert "majority sample rate" in joined

    def test_aligned_output_is_quiet(self) -> None:
        intent = IntentCategories(sample_rate=48000, duration_max_sec=60.0, duration_min_sec=2.0)
        out_card = DatasetCard(
            name="out", uri="y",
            profile=DatasetProfile(
                total_files=10,
                decodable_files=10,
                sample_rates_hz={"48000": 10},
                duration_p05_sec=2.5,
                duration_p50_sec=10.0,
                duration_p95_sec=55.0,
            ),
        )
        rep = critique(run_card=_run_card([]), intent=intent, output_card=out_card)
        assert rep.deterministic_pass is True
