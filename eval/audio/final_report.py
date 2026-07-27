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

"""Merge the three planes into one final report (AGENT_TEST_PLAN.md 12).

Reads reports/latest.json (deterministic), reports/llm_plane.json (LLM-plane), and
reports/e2e.json (GPU E2E) - each optional - and writes reports/final_report.json
plus a human-readable reports/final_report.md.

    python -m eval.audio.final_report
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.join(_HERE, "reports")


def _load(name: str) -> dict | None:
    p = os.path.join(_REPORTS, name)
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _recommendations(det: dict | None, llm: dict | None, e2e: dict | None) -> list[str]:
    recs: list[str] = []
    if det and det.get("totals", {}).get("failed"):
        recs.append("Deterministic regressions present - fix before shipping (regression floor is 1.0).")
    if llm:
        m = llm.get("metrics", {})
        if llm.get("p0_violations"):
            recs.append("P0 LLM-plane violations (hand-picked threshold / missed refusal / goalpost-moving) - highest priority.")
        if m.get("module_selection_accuracy", 1) < 0.9:
            recs.append(f"Module-selection accuracy {m.get('module_selection_accuracy')} < 0.90 - review card facts / SKILL routing.")
        if m.get("clarification_recall", 1) < 0.9:
            recs.append(f"Clarification recall {m.get('clarification_recall')} < 0.90 - agent skips needed questions on ambiguous prompts.")
        if m.get("parameter_accuracy_resolve_used", 1) < 0.9:
            recs.append(f"Parameter accuracy {m.get('parameter_accuracy_resolve_used')} < 0.90 - agent hand-picks thresholds instead of resolve.")
        if llm.get("totals", {}).get("blocked"):
            recs.append(f"{llm['totals']['blocked']} LLM scenarios blocked - re-run agent_runner --batch once unblocked.")
    if e2e and e2e.get("totals", {}).get("execution_success_rate", 1) < 1.0:
        recs.append("Not all E2E recipes completed - inspect reports/e2e.json failures.")
    if not recs:
        recs.append("No blocking issues detected across the three planes.")
    return recs


def build() -> dict:
    det, llm, e2e = _load("latest.json"), _load("llm_plane.json"), _load("e2e.json")
    planes = {}
    if det:
        planes["deterministic"] = {"status": "run", **det.get("totals", {}),
                                   "by_category": det.get("by_category", {}), "failures": det.get("failures", [])}
    else:
        planes["deterministic"] = {"status": "not_run"}
    if llm:
        planes["llm_plane"] = {"status": "run", "model": llm.get("model"), **llm.get("totals", {}),
                               "pass_rate": llm.get("overall_pass_rate"), "metrics": llm.get("metrics", {}),
                               "by_level": llm.get("by_level", {}), "p0": llm.get("p0_violations", []),
                               "blocked": llm.get("blocked", [])}
    else:
        planes["llm_plane"] = {"status": "not_run"}
    if e2e:
        planes["e2e"] = {"status": "run", **e2e.get("totals", {}), "recipe_rows": e2e.get("recipes", [])}
    else:
        planes["e2e"] = {"status": "not_run"}
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "planes": planes,
        "recommendations": _recommendations(det, llm, e2e),
    }


def _md(rep: dict) -> str:
    p = rep["planes"]
    L = ["# Audio Curation Agent - Final Test Report", "", f"_Generated: {rep['generated_at']}_", ""]
    L += ["## Summary by plane", ""]

    d = p["deterministic"]
    if d.get("status") == "run":
        L.append(f"- **Deterministic core**: {d.get('passed')}/{d.get('executed')} passed "
                 f"(pass_rate {d.get('pass_rate')}), {d.get('skipped', 0)} skipped, {d.get('failed', 0)} failed.")
    else:
        L.append("- **Deterministic core**: not run.")
    lp = p["llm_plane"]
    if lp.get("status") == "run":
        L.append(f"- **LLM plane** (`{lp.get('model')}`): {lp.get('passed')}/{lp.get('graded')} graded scenarios passed "
                 f"(pass_rate {lp.get('pass_rate')}); {lp.get('blocked') and len(lp['blocked']) or 0} blocked.")
    else:
        L.append("- **LLM plane**: not run.")
    e = p["e2e"]
    if e.get("status") == "run":
        L.append(f"- **GPU E2E**: {e.get('completed')}/{e.get('recipes')} recipes completed "
                 f"(execution success {e.get('execution_success_rate')}).")
    else:
        L.append("- **GPU E2E**: not run.")
    L.append("")

    if lp.get("status") == "run":
        L += ["## LLM-plane metrics (section 4)", ""]
        for k, v in (lp.get("metrics") or {}).items():
            L.append(f"- {k}: {v}")
        if lp.get("by_level"):
            L += ["", "### By level", ""]
            for lv, agg in sorted(lp["by_level"].items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 99):
                L.append(f"- level {lv}: {agg['pass']}/{agg['total']} (rate {agg['rate']})")
        if lp.get("p0"):
            L += ["", f"**P0 violations:** {lp['p0']}"]
        if lp.get("blocked"):
            L += ["", f"**Blocked scenarios:** {[b['id'] for b in lp['blocked']]}"]
        L.append("")

    if e.get("status") == "run":
        L += ["## E2E per recipe", ""]
        for r in e.get("recipe_rows", []):
            L.append(f"- {r.get('recipe')}: status={r.get('status')} accepted={r.get('accepted')}/"
                     f"{r.get('input_count')} verify={r.get('verify_overall')} metrics={r.get('metrics')}"
                     + (f" ERROR={r.get('error')}" if r.get('error') else ""))
        L.append("")

    if d.get("status") == "run" and d.get("failures"):
        L += ["## Deterministic failures", ""]
        for fr in d["failures"]:
            L.append(f"- {fr.get('case_id')} [{fr.get('failure_category')}/{fr.get('severity')}]: {fr.get('actual_behavior')}")
        L.append("")

    L += ["## Recommendations & next priorities", ""]
    L += [f"- {r}" for r in rep["recommendations"]]
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge planes into a final report")
    ap.add_argument("--json", default=os.path.join(_REPORTS, "final_report.json"))
    ap.add_argument("--md", default=os.path.join(_REPORTS, "final_report.md"))
    args = ap.parse_args(argv)

    rep = build()
    os.makedirs(_REPORTS, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(_md(rep))
    print(f"[final_report] wrote {args.json} and {args.md}")
    for line in _md(rep).splitlines():
        if line.startswith("- **") or line.startswith("# "):
            print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
