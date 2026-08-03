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

    python -m eval.audio.aggregate_traces --non-llm
    python -m eval.audio.aggregate_traces --judge-model claude-opus-4-8
    python -m eval.audio.aggregate_traces --report eval/audio/reports/llm_plane.json

``--non-llm`` is an explicit diagnostic mode and never claims semantic
certification. ``--judge-model`` requires complete, current Level-15 coverage;
missing, blocked, structurally failing, non-authoritative, or model-judged misses
all fail the regression gate.
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


def _load_scenarios() -> dict[str, dict]:
    import yaml

    scenarios: dict[str, dict] = {}
    for path in glob.glob(os.path.join(_HERE, "scenarios", "*.yaml")):
        with open(path, encoding="utf-8") as stream:
            for scenario in (yaml.safe_load(stream) or {}).get("scenarios", []):
                scenarios[scenario["id"]] = scenario
    return scenarios


def _semantic_scenario_ids(scenarios: dict[str, dict]) -> list[str]:
    return sorted(
        sid
        for sid, scenario in scenarios.items()
        if scenario.get("semantic_expectations")
    )


def _collect(*, judge_model=None, traces_dir: str = _TRACES_DIR) -> list[dict]:  # noqa: ANN001
    from eval.audio import judge as judge_mod
    from eval.audio import trace_check as tc

    scenarios = _load_scenarios()

    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(traces_dir, "*.json"))):
        base = os.path.basename(path)
        if ".good." in base or ".bad." in base:
            continue  # bundled grading examples, not scenario runs
        with open(path, encoding="utf-8") as stream:
            trace = json.load(stream)
        sid = trace.get("scenario_id") or os.path.splitext(base)[0]
        scenario = scenarios.get(sid)
        if not scenario:
            continue
        if trace.get("blocked"):
            rows.append({"id": sid, "level": scenario.get("level"), "status": "blocked",
                         "error": trace.get("error"), "trace": trace, "scenario": scenario})
            continue
        card = tc.grade(scenario, trace)
        semantic = bool(scenario.get("semantic_expectations"))
        jr = judge_mod.judge(
            scenario,
            trace,
            model=judge_model if semantic else None,
        )
        status = card["overall"]
        violations = tc.classify_violations(card["violations"])
        semantic_judge_pass: bool | None = None
        if semantic and jr.get("authoritative"):
            threshold = float(
                (scenario.get("semantic_expectations") or {}).get(
                    "judge_min_score", 0.8
                )
            )
            semantic_judge_pass = bool(
                jr.get("grounded") is True
                and not isinstance(jr.get("overall"), bool)
                and isinstance(jr.get("overall"), (int, float))
                and 0.0 <= jr["overall"] <= 1.0
                and jr["overall"] >= threshold
            )
            if not semantic_judge_pass:
                status = "fail"
                violations.extend(
                    tc.classify_violations(["intent_misunderstanding"])
                )
        rows.append({
            "id": sid, "level": scenario.get("level"), "status": status,
            "deterministic_status": card["overall"],
            "dimensions": {k: bool(v.get("pass")) for k, v in card["dimensions"].items()},
            "violations": violations,
            "asked": tc._asked_clarification(trace),
            "clar_required": bool(scenario.get("clarification_required") or scenario.get("expect_clarification")),
            "judge": jr.get("overall"), "grounded": jr.get("grounded"),
            "judge_authoritative": bool(jr.get("authoritative")),
            "judge_validation_error": jr.get("validation_error"),
            "semantic_judge_pass": semantic_judge_pass,
            "captured_by": trace.get("captured_by"),
        })
    return rows


def _semantic_gate(
    rows: list[dict],
    *,
    expected_ids: list[str],
    authoritative_required: bool,
) -> dict:
    """Fail closed when an authoritative semantic campaign is incomplete."""
    by_id = {str(row.get("id")): row for row in rows}
    definition_missing = authoritative_required and not expected_ids
    missing = sorted(sid for sid in expected_ids if sid not in by_id)
    blocked = sorted(
        sid
        for sid in expected_ids
        if sid in by_id and by_id[sid].get("status") == "blocked"
    )
    deterministic_failures = sorted(
        sid
        for sid in expected_ids
        if sid in by_id
        and by_id[sid].get("deterministic_status") != "pass"
        and by_id[sid].get("status") != "blocked"
    )
    non_authoritative = sorted(
        sid
        for sid in expected_ids
        if sid in by_id
        and by_id[sid].get("status") != "blocked"
        and by_id[sid].get("judge_authoritative") is not True
    )
    judge_failures = sorted(
        sid
        for sid in expected_ids
        if sid in by_id
        and by_id[sid].get("status") != "blocked"
        and by_id[sid].get("semantic_judge_pass") is not True
    )
    passed: bool | None = None
    if authoritative_required:
        passed = not any(
            (
                missing,
                blocked,
                deterministic_failures,
                non_authoritative,
                judge_failures,
                definition_missing,
            )
        )
    return {
        "mode": "authoritative_model" if authoritative_required else "non_llm_diagnostic",
        "required": authoritative_required,
        "expected": list(expected_ids),
        "definition_missing": definition_missing,
        "observed": sorted(sid for sid in expected_ids if sid in by_id),
        "missing": missing,
        "blocked": blocked,
        "deterministic_failures": deterministic_failures,
        "non_authoritative": non_authoritative,
        "judge_failures": judge_failures,
        "passed": passed,
    }


def build(
    rows: list[dict],
    *,
    required_semantic_ids: list[str] | None = None,
    authoritative_required: bool = False,
) -> dict:
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
    semantic_rows = [
        r for r in graded if r.get("semantic_judge_pass") is not None
    ]
    metrics = {
        "module_selection_accuracy": dim_rate("modules"),
        "pipeline_validity": dim_rate("compatible"),
        "parameter_accuracy_resolve_used": dim_rate("configured"),
        "completeness": dim_rate("completeness"),
        "semantic_critique_accuracy": dim_rate("semantic_critique"),
        "refusal_accuracy": dim_rate("refusal"),
        "clarification_precision": _rate(tp, tp + fp),
        "clarification_recall": _rate(tp, tp + fn),
        "explanation_mean": round(sum(judges) / len(judges), 3) if judges else 0.0,
        "grounded_rate": _rate(sum(1 for r in graded if r.get("grounded")), len(graded)),
        "semantic_authoritative_judged": len(semantic_rows),
        "semantic_authoritative_pass_rate": _rate(
            sum(1 for r in semantic_rows if r.get("semantic_judge_pass")),
            len(semantic_rows),
        ),
    }

    p0 = [v for r in graded for v in (r.get("violations") or []) if v.get("severity") == "P0"]
    model = next((r.get("captured_by") for r in rows if r.get("captured_by")), None)
    semantic_gate = _semantic_gate(
        rows,
        expected_ids=list(required_semantic_ids or []),
        authoritative_required=authoritative_required,
    )
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model": model,
        "totals": {"scenarios": len(rows), "graded": len(graded), "passed": len(passed),
                   "failed": len(graded) - len(passed), "blocked": len(blocked)},
        "overall_pass_rate": _rate(len(passed), len(graded)),
        "by_dimension": dims,
        "by_level": levels,
        "metrics": metrics,
        "semantic_gate": semantic_gate,
        "p0_violations": p0,
        "blocked": [{"id": r["id"], "error": r.get("error")} for r in blocked],
        "scenarios": [{k: r.get(k) for k in ("id", "level", "status", "deterministic_status", "dimensions", "violations", "judge", "grounded", "judge_authoritative", "judge_validation_error", "semantic_judge_pass")}
                      for r in rows if r["status"] != "blocked"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate LLM-plane traces")
    ap.add_argument("--report", default=_DEFAULT_REPORT)
    ap.add_argument("--traces-dir", default=_TRACES_DIR)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--judge-model",
        help="Cursor model id; authoritatively judge and gate semantic scenarios",
    )
    mode.add_argument(
        "--non-llm",
        action="store_true",
        help="diagnostic-only fallback; does not certify or gate semantic correctness",
    )
    args = ap.parse_args(argv)

    from eval.audio import judge as judge_mod

    model = judge_mod.cursor_model(args.judge_model) if args.judge_model else None
    scenarios = _load_scenarios()
    semantic_ids = _semantic_scenario_ids(scenarios)
    rep = build(
        _collect(judge_model=model, traces_dir=args.traces_dir),
        required_semantic_ids=semantic_ids,
        authoritative_required=bool(args.judge_model),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    t = rep["totals"]
    print(f"[llm_plane] graded={t['graded']} passed={t['passed']} failed={t['failed']} "
          f"blocked={t['blocked']} pass_rate={rep['overall_pass_rate']}")
    print(f"[llm_plane] metrics={json.dumps(rep['metrics'])}")
    print(f"[llm_plane] semantic_gate={json.dumps(rep['semantic_gate'])}")
    print(f"[llm_plane] wrote {args.report}")
    if args.non_llm:
        print("[llm_plane] NON-LLM DIAGNOSTIC ONLY: semantic correctness is not certified")
        return 0
    return 0 if rep["semantic_gate"]["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
