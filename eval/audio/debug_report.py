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

"""Consolidated inside-out debug report (AGENT_TEST_PLAN.md Part 3).

Renders reports/debug_report.md from all available artifacts so every run is
inspectable end to end: the components/flow reference, the deterministic suite,
the LLM-plane scenario traces, and the end-to-end user-simulation personas, plus
a Findings (fix later) section.

    python -m eval.audio.debug_report
"""

from __future__ import annotations

import argparse
import glob
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.join(_HERE, "reports")
_TRACES = os.path.join(_HERE, "traces")
_SIM_TRACES = os.path.join(_TRACES, "sim")


def _load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _components_reference() -> list[str]:
    return [
        "## Components & flow reference",
        "",
        "The agent is two planes: an LLM **planner/critic** (host model following "
        "`.claude/skills/audio-curation/SKILL.md`) and a deterministic **core** "
        "(`nemo_curator.audio_agent`, exposed as CLI + MCP tools). The LLM proposes; "
        "the core grounds and checks every decision.",
        "",
        "Core verbs:",
        "- `discover` / `catalog_tree` / `cards` - find capabilities (stages) and read their spec cards.",
        "- `context` - profile the dataset + machine, match blueprints.",
        "- `resolve` - map a plain outcome ('studio') to a concrete parameter (no hand-picked numbers).",
        "- `validate` - check a recipe (verdict pass/fail/uncertain + issue codes).",
        "- `smoke` - tiny trial run; returns config_hash + smoke_token + calibration.",
        "- `run` - confirm-gated full run (0 silent runs).",
        "- `report` / `verify` - summarize results and prove the acceptance contract.",
        "- `plan_continuation` / `calibrate` - reuse prior work; refine resource estimates.",
        "",
        "9-step flow: intent -> inspect -> route (L0/L1/L2) -> resolve -> validate loop -> "
        "smoke -> confirm gate -> run -> report + verify (+ continuation for follow-ups).",
        "",
    ]


def _deterministic_section() -> list[str]:
    det = _load(os.path.join(_REPORTS, "latest.json"))
    L = ["## Deterministic suite", ""]
    if not det:
        return L + ["_not run (reports/latest.json missing)_", ""]
    t = det.get("totals", {})
    L.append(f"Totals: executed={t.get('executed')} passed={t.get('passed')} failed={t.get('failed')} "
             f"skipped={t.get('skipped')} pass_rate={t.get('pass_rate')}.")
    L.append("")
    bc = det.get("by_category", {})
    if bc:
        L.append("By category: " + ", ".join(f"{k} {v['passed']}/{v['total']}" for k, v in sorted(bc.items())))
        L.append("")
    if det.get("failures"):
        L.append("Failures:")
        for f in det["failures"]:
            L.append(f"- {f.get('case_id')} [{f.get('failure_category')}/{f.get('severity')}]: {f.get('actual_behavior')}")
        L.append("")
    return L


def _llm_plane_section() -> list[str]:
    lp = _load(os.path.join(_REPORTS, "llm_plane.json"))
    L = ["## LLM-plane scenarios (inside-out)", ""]
    if not lp:
        return L + ["_not run (reports/llm_plane.json missing)_", ""]
    t = lp.get("totals", {})
    L.append(f"Model: `{lp.get('model')}` - graded={t.get('graded')} passed={t.get('passed')} "
             f"failed={t.get('failed')} blocked={t.get('blocked')} pass_rate={lp.get('overall_pass_rate')}.")
    L.append("")
    for s in lp.get("scenarios", []):
        sid = s.get("id")
        tr = _load(os.path.join(_TRACES, f"{sid}.json")) or {}
        verbs = [c.get("verb") for c in tr.get("tool_calls", []) if c.get("verb")]
        refs = [x.get("ref") for x in (tr.get("final_recipe") or {}).get("stages", [])]
        dims = s.get("dimensions", {})
        fails = [k for k, v in dims.items() if not v]
        L.append(f"### {sid} - {s.get('status', '?').upper()}")
        if tr.get("prompt"):
            L.append(f"- prompt: {tr['prompt']}")
        L.append(f"- tool flow: {' -> '.join(verbs) if verbs else '(none)'}")
        L.append(f"- built recipe: {refs or '(none / deferred)'}")
        L.append(f"- grade: {'PASS' if not fails else 'FAIL (' + ','.join(fails) + ')'}")
        if tr.get("explanation"):
            L.append(f"- rationale: {tr['explanation'][:220].strip()}")
        L.append("")
    return L


def _persona_section() -> list[str]:
    e2e = _load(os.path.join(_REPORTS, "e2e_from_prompt.json"))
    L = ["## End-to-end personas (user-simulation)", ""]
    if not e2e:
        return L + ["_not run (reports/e2e_from_prompt.json missing)_", ""]
    t = e2e.get("totals", {})
    L.append(f"SUT=`{e2e.get('model_sut')}` USER=`{e2e.get('model_user')}` - personas={t.get('personas')} "
             f"passed={t.get('passed')} failed={t.get('failed')} blocked={t.get('blocked')} pass_rate={t.get('pass_rate')}.")
    L.append("")
    for p in e2e.get("personas", []):
        pid = p.get("id")
        L.append(f"### {pid} ({p.get('group')}) - {str(p.get('grade')).upper()}")
        L.append(f"- prompt: {p.get('opening_prompt')}")
        # transcript (abbreviated) from the sim trace
        tr = _load(os.path.join(_SIM_TRACES, f"{pid}.json")) or {}
        convo = (tr.get("conversation") or {}).get("transcript", [])
        for turn in convo:
            who = turn.get("speaker", "?")
            txt = (turn.get("text") or "").strip().replace("\n", " ")
            if txt:
                L.append(f"  - {who}: {txt[:200]}")
            tcs = [c.get("verb") for c in turn.get("tool_calls", []) if c.get("verb")]
            if tcs:
                L.append(f"    tools: {' -> '.join(tcs)}")
        L.append(f"- outcome: {p.get('outcome')} | asked_clarification: {p.get('asked_clarification')}")
        L.append(f"- built recipe: {p.get('final_recipe') or '(none)'}")
        if p.get("execution"):
            ex = p["execution"]
            L.append(f"- execution: status={ex.get('status')} accepted={ex.get('accepted')}/{ex.get('input_count')} "
                     f"verify={ex.get('verify_overall')} metrics={ex.get('metrics')}"
                     + (f" ERROR={ex.get('error')}" if ex.get("error") else ""))
        dims = p.get("dimensions", {})
        L.append("- grade: " + ", ".join(f"{k}={'ok' if v.get('pass') else 'FAIL'}" for k, v in dims.items()))
        if p.get("findings"):
            L.append(f"- findings: {[f'{f['stage']}/{f['severity']}' for f in p['findings']]}")
        L.append("")
    return L


def _findings_section() -> list[str]:
    fnd = _load(os.path.join(_REPORTS, "findings.json"))
    L = ["## Findings (fix later)", ""]
    if not fnd or not fnd.get("findings"):
        return L + ["_none recorded_", ""]
    L.append(f"{fnd.get('count')} findings; by severity: {fnd.get('by_severity')}. Nothing fixed in this run.")
    L.append("")
    for f in sorted(fnd["findings"], key=lambda x: x.get("severity", "P3")):
        L.append(f"- **{f.get('severity')}** [{f.get('category')} @ {f.get('stage')}] {f.get('source')}: "
                 f"{f.get('what_happened')} -> _fix:_ {f.get('suggested_fix')}")
    L.append("")
    return L


def build() -> str:
    import datetime as _dt

    L = ["# Audio Curation Agent - Inside-Out Debug Report", "",
         f"_Generated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}_", ""]
    L += _components_reference()
    L += _deterministic_section()
    L += _llm_plane_section()
    L += _persona_section()
    L += _findings_section()
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the inside-out debug report")
    ap.add_argument("--out", default=os.path.join(_REPORTS, "debug_report.md"))
    args = ap.parse_args(argv)
    os.makedirs(_REPORTS, exist_ok=True)
    md = build()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[debug_report] wrote {args.out} ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
