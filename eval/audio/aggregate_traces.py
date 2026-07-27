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

"""Aggregate captured LLM-plane traces into reports/llm_plane.json (AGENT_TEST_PLAN 4).

Grades every trace under traces/ (except the *.good./*.bad. examples) with
trace_check + judge, rolls up per-dimension and per-level pass rates, and computes
the section-4 LLM-plane metrics. Blocked traces (SDK auth/config failures) are
counted separately, not as failures.

    python -m eval.audio.aggregate_traces
    python -m eval.audio.aggregate_traces --report eval/audio/reports/llm_plane.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRACES_DIR = os.path.join(_HERE, "traces")
_DEFAULT_REPORT = os.path.join(_HERE, "reports", "llm_plane.json")


def _rate(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def _collect() -> list[dict]:
    from eval.audio import judge as judge_mod
    from eval.audio import trace_check as tc

    scenarios = {}
    import glob as _g

    import yaml
    for f in _g.glob(os.path.join(_HERE, "scenarios", "*.yaml")):
        for s in (yaml.safe_load(open(f, encoding="utf-8")) or {}).get("scenarios", []):
            scenarios[s["id"]] = s

    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(_TRACES_DIR, "*.json"))):
        base = os.path.basename(path)
        if ".good." in base or ".bad." in base:
            continue  # bundled grading examples, not scenario runs
        trace = json.load(open(path, encoding="utf-8"))
        sid = trace.get("scenario_id") or os.path.splitext(base)[0]
        scenario = scenarios.get(sid)
        if not scenario:
            continue
        if trace.get("blocked"):
            rows.append({"id": sid, "level": scenario.get("level"), "status": "blocked",
                         "error": trace.get("error"), "trace": trace, "scenario": scenario})
            continue
        card = tc.grade(scenario, trace)
        jr = judge_mod.judge(scenario, trace)
        rows.append({
            "id": sid, "level": scenario.get("level"), "status": card["overall"],
            "dimensions": {k: bool(v.get("pass")) for k, v in card["dimensions"].items()},
            "violations": tc.classify_violations(card["violations"]),
            "asked": tc._asked_clarification(trace),
            "clar_required": bool(scenario.get("clarification_required") or scenario.get("expect_clarification")),
            "judge": jr.get("overall"), "grounded": jr.get("grounded"),
            "captured_by": trace.get("captured_by"),
        })
    return rows


def build(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["status"] in ("pass", "fail")]
    blocked = [r for r in rows if r["status"] == "blocked"]
    passed = [r for r in graded if r["status"] == "pass"]

    # per-dimension
    dims: dict[str, dict] = {}
    for r in graded:
        for d, ok in (r.get("dimensions") or {}).items():
            agg = dims.setdefault(d, {"pass": 0, "total": 0})
            agg["total"] += 1
            agg["pass"] += 1 if ok else 0
    for d in dims:
        dims[d]["rate"] = _rate(dims[d]["pass"], dims[d]["total"])

    # per-level
    levels: dict[str, dict] = {}
    for r in graded:
        lv = str(r.get("level"))
        agg = levels.setdefault(lv, {"pass": 0, "total": 0})
        agg["total"] += 1
        agg["pass"] += 1 if r["status"] == "pass" else 0
    for lv in levels:
        levels[lv]["rate"] = _rate(levels[lv]["pass"], levels[lv]["total"])

    # clarification precision/recall
    tp = sum(1 for r in graded if r["asked"] and r["clar_required"])
    fp = sum(1 for r in graded if r["asked"] and not r["clar_required"])
    fn = sum(1 for r in graded if not r["asked"] and r["clar_required"])

    def dim_rate(name: str) -> float:
        d = dims.get(name)
        return d["rate"] if d else 0.0

    judges = [r["judge"] for r in graded if isinstance(r.get("judge"), (int, float))]
    metrics = {
        "module_selection_accuracy": dim_rate("modules"),
        "pipeline_validity": dim_rate("compatible"),
        "parameter_accuracy_resolve_used": dim_rate("configured"),
        "completeness": dim_rate("completeness"),
        "refusal_accuracy": dim_rate("refusal"),
        "clarification_precision": _rate(tp, tp + fp),
        "clarification_recall": _rate(tp, tp + fn),
        "explanation_mean": round(sum(judges) / len(judges), 3) if judges else 0.0,
        "grounded_rate": _rate(sum(1 for r in graded if r.get("grounded")), len(graded)),
    }

    p0 = [v for r in graded for v in (r.get("violations") or []) if v.get("severity") == "P0"]
    model = next((r.get("captured_by") for r in rows if r.get("captured_by")), None)
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model": model,
        "totals": {"scenarios": len(rows), "graded": len(graded), "passed": len(passed),
                   "failed": len(graded) - len(passed), "blocked": len(blocked)},
        "overall_pass_rate": _rate(len(passed), len(graded)),
        "by_dimension": dims,
        "by_level": levels,
        "metrics": metrics,
        "p0_violations": p0,
        "blocked": [{"id": r["id"], "error": r.get("error")} for r in blocked],
        "scenarios": [{k: r.get(k) for k in ("id", "level", "status", "dimensions", "violations", "judge", "grounded")}
                      for r in rows if r["status"] != "blocked"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate LLM-plane traces")
    ap.add_argument("--report", default=_DEFAULT_REPORT)
    args = ap.parse_args(argv)

    rep = build(_collect())
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    t = rep["totals"]
    print(f"[llm_plane] graded={t['graded']} passed={t['passed']} failed={t['failed']} "
          f"blocked={t['blocked']} pass_rate={rep['overall_pass_rate']}")
    print(f"[llm_plane] metrics={json.dumps(rep['metrics'])}")
    print(f"[llm_plane] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
