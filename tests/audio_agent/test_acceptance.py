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

"""Unit tests for the acceptance/honesty gate (verify)."""

from nemo_curator import audio_agent as aa


def _crit(cid: str, field: str, op: str, value: float, ctype: str = "quality_standard", severity: str = "must") -> dict:
    return {"id": cid, "type": ctype, "severity": severity, "check": {"field": field, "op": op, "value": value}}


class TestVerify:
    def test_must_met_when_metric_satisfies(self) -> None:
        r = aa.verify([_crit("q", "mos", ">=", 3.0)], evidence={"metrics": {"mos": 3.5}})
        assert r["overall"] == "met"

    def test_must_not_met_when_metric_below_bar(self) -> None:
        r = aa.verify([_crit("q", "mos", ">=", 3.0)], evidence={"metrics": {"mos": 2.0}})
        assert r["overall"] == "not_met"

    def test_no_evidence_is_not_silently_met(self) -> None:
        r = aa.verify([_crit("q", "mos", ">=", 3.0)], evidence={})
        assert r["overall"] == "not_met"
        assert any(c.get("status") == "unverifiable" for c in r.get("criteria", []))

    def test_empty_contract_is_unverifiable_not_met(self) -> None:
        # No criteria at all -> nothing was verified: never a silent "met", but also not
        # "not_met" (which would over-reject a run that carried no explicit success bar).
        r = aa.verify([], evidence={"metrics": {"mos": 3.5}, "retained": 10, "input_count": 10})
        assert r["overall"] == "unverifiable"

    def test_nice_only_contract_stays_met(self) -> None:
        # A non-empty contract with only 'nice' criteria remains met (nice = non-blocking).
        r = aa.verify([_crit("q", "mos", ">=", 3.0, severity="nice")], evidence={"metrics": {"mos": 2.0}})
        assert r["overall"] == "met"


class TestHonestyGuard:
    def test_relaxed_must_forces_not_met(self) -> None:
        frozen = [_crit("q", "mos", ">=", 4.0)]
        used = [_crit("q", "mos", ">=", 3.0)]  # easier bar than confirmed
        r = aa.verify(used, evidence={"metrics": {"mos": 3.5}}, frozen_criteria=frozen)
        assert r["overall"] == "not_met"
        assert any(h.get("code") == "must_relaxed" for h in r.get("honesty", []))

    def test_dropped_must_flagged(self) -> None:
        frozen = [_crit("q", "mos", ">=", 3.0)]
        r = aa.verify([], evidence={"metrics": {"mos": 3.5}}, frozen_criteria=frozen)
        assert any(h.get("code") == "must_dropped" for h in r.get("honesty", []))
