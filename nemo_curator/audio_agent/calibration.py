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

"""Resource calibration (1C.2): measured per-stage resources refine the plan.

A smoke run observes what each stage actually used; :func:`from_smoke` extracts
those measurements so the resource planner can prefer **measured** numbers over a
card's ``best_guess`` ``resource`` facts on the next plan. This is the plumbing —
the *measurements themselves* come from a real (GPU) smoke; on a CPU smoke there is
no VRAM to read, so extraction is empty and the planner falls back to card facts.

Nothing here is shared across users/sessions: a calibration is derived from *this*
smoke and applies to *this* machine (stamped with the machine fingerprint), i.e. an
operational, recomputable annotation — consistent with the no-memory stance.
"""

from __future__ import annotations

from typing import Any

# Candidate per-stage perf-metric names a smoke may expose for each resource fact.
# (StagePerfStats keys vary by backend/instrumentation; we read whichever is present.)
_VRAM_KEYS = ("peak_vram_gb", "gpu_mem_gb", "vram_gb", "custom.peak_vram_gb", "custom.gpu_mem_gb")
_HOST_MEM_KEYS = ("peak_host_mem_gb", "host_mem_gb", "rss_gb", "custom.peak_host_mem_gb")
_THROUGHPUT_KEYS = ("throughput", "items_per_sec", "custom.throughput")


def _read_metric(metrics: dict[str, Any], keys: tuple[str, ...], *, prefer: str = "max") -> float | None:
    """Read the first present metric among ``keys``.

    ``per_stage_metrics`` values are ``{sum, mean, count}`` aggregates (peak/max is
    not currently captured), so ``prefer`` falls back mean -> sum. A bare number is
    taken as-is.
    """
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, dict):
            for stat in (prefer, "mean", "sum"):
                if stat in v and isinstance(v[stat], (int, float)):
                    return float(v[stat])
        elif isinstance(v, (int, float)):
            return float(v)
    return None


def from_smoke(smoke_report: dict[str, Any] | None, *, machine_fingerprint: str | None = None) -> dict[str, Any]:
    """Measured per-stage resource facts from a smoke report's ``per_stage_metrics``.

    Returns ``{stage: {gpu_mem_gb?, host_mem_gb?, throughput?, source: "measured",
    machine_fingerprint?}}`` for stages that carry a reading; empty when the perf
    stats have no resource measurements (e.g. a CPU smoke). Feed to
    ``run(..., calibration=...)`` so the planner uses measured over card numbers.
    """
    per_stage = (smoke_report or {}).get("per_stage_metrics") or {}
    out: dict[str, Any] = {}
    for stage, metrics in per_stage.items():
        if not isinstance(metrics, dict):
            continue
        entry: dict[str, Any] = {}
        vram = _read_metric(metrics, _VRAM_KEYS)
        if vram is not None:
            entry["gpu_mem_gb"] = round(vram, 3)
        host = _read_metric(metrics, _HOST_MEM_KEYS)
        if host is not None:
            entry["host_mem_gb"] = round(host, 3)
        thr = _read_metric(metrics, _THROUGHPUT_KEYS, prefer="mean")
        if thr is not None:
            entry["throughput"] = round(thr, 3)
        if entry:
            entry["source"] = "measured"
            if machine_fingerprint:
                entry["machine_fingerprint"] = machine_fingerprint
            out[stage] = entry
    return out
