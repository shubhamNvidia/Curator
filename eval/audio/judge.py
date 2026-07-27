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

"""LLM-as-judge for the qualitative dimensions of an agent trace (explanation
grounding, module-selection justification, clarification quality).

Two modes:

* **Model mode** — pass a callable ``model(prompt) -> str`` (e.g. wrapping the
  Cursor SDK or any chat API). The judge sends a rubric prompt and parses the
  JSON verdict. This is the intended production path.
* **Deterministic fallback** (default) — no model needed. Scores each rubric item
  by keyword grounding against the captured ``explanation`` and flags claims not
  supported by the trace's tool results. Keeps the harness runnable in CI.

    python -m eval.audio.judge --id L02_readspeech_quality --trace traces/L02.json
    python -m eval.audio.judge --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

_STOP = set(
    "the a an of to and or is are be for with without it its this that as on in into per "
    "not no you your my their via from up down over under only just came come each was were "
    "did does do so than then them they there here what which who when where why how".split()
)
_NUM = re.compile(r"\d+(?:\.\d+)?")
_PASS_ITEM = 0.34   # an item is "met" when at least this fraction of its key terms are grounded


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z_]+", (text or "").lower()) if w not in _STOP and len(w) > 2}


def _trace_result_text(trace: dict) -> str:
    """All numbers/text the agent actually observed (for grounding claims)."""
    return json.dumps([tc.get("result") for tc in trace.get("tool_calls", []) if tc.get("result") is not None])


def _deterministic(scenario: dict, trace: dict) -> dict:
    rubric = scenario.get("explanation_rubric", []) or []
    expl = (trace.get("explanation") or "").strip()
    expl_kw = _keywords(expl)
    items = []
    for r in rubric:
        kw = _keywords(r)
        score = 1.0 if not kw else round(len(kw & expl_kw) / len(kw), 3)
        items.append({"criterion": r, "score": score, "met": score >= _PASS_ITEM})
    # grounding: any number claimed in the explanation should appear in an observed result
    observed = _trace_result_text(trace)
    claimed_nums = set(_NUM.findall(expl))
    observed_nums = set(_NUM.findall(observed))
    ungrounded = sorted(n for n in claimed_nums if n not in observed_nums)
    overall = round(sum(i["score"] for i in items) / len(items), 3) if items else (1.0 if expl else 0.0)
    return {
        "mode": "deterministic",
        "authoritative": False,  # keyword overlap + number-grounding, NOT a semantic quality score
        "scenario_id": scenario.get("id"),
        "items": items,
        "overall": overall,
        "grounded": not ungrounded,
        "ungrounded_numbers": ungrounded,
        "note": "keyword-grounding fallback (indicative only, not a semantic quality score); pass model= for real judgement",
    }


def _model_judge(scenario: dict, trace: dict, model) -> dict:  # noqa: ANN001
    prompt = (
        "You are grading an audio-curation agent's explanation against a rubric. "
        "Return STRICT JSON: {\"items\":[{\"criterion\":str,\"score\":0..1,\"met\":bool}],"
        "\"overall\":0..1,\"grounded\":bool,\"notes\":str}.\n\n"
        f"PROMPT: {scenario.get('prompt')}\n"
        f"RUBRIC: {json.dumps(scenario.get('explanation_rubric', []))}\n"
        f"AGENT EXPLANATION: {trace.get('explanation', '')}\n"
        f"OBSERVED TOOL RESULTS: {_trace_result_text(trace)}\n"
    )
    raw = model(prompt)
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - be forgiving; fall back to extracting the JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {"overall": 0.0, "items": [], "grounded": False}
    data["mode"] = "model"
    data["authoritative"] = True
    data["scenario_id"] = scenario.get("id")
    return data


def judge(scenario: dict, trace: dict, *, model=None) -> dict:  # noqa: ANN001
    """Score the qualitative rubric. Uses ``model`` if given, else the fallback."""
    return _model_judge(scenario, trace, model) if model else _deterministic(scenario, trace)


def _selftest() -> int:
    from eval.audio.trace_check import _BAD_TRACE, _GOOD_TRACE, find_scenario

    scenario = find_scenario("L02_readspeech_quality")
    good = judge(scenario, _GOOD_TRACE)
    bad = judge(scenario, _BAD_TRACE)
    print(f"[selftest] good overall={good['overall']} grounded={good['grounded']} | "
          f"bad overall={bad['overall']} grounded={bad['grounded']}")
    ok = good["overall"] > bad["overall"] and good["grounded"]
    print("[selftest]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge for agent explanations")
    ap.add_argument("--scenario", help="path to a scenarios/*.yaml file")
    ap.add_argument("--id", help="scenario id")
    ap.add_argument("--trace", help="path to a captured trace JSON")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.trace:
        ap.error("--trace is required (or use --selftest)")

    from eval.audio.trace_check import find_scenario, load_scenario

    with open(args.trace, encoding="utf-8") as f:
        trace = json.load(f)
    scenario = load_scenario(args.scenario, args.id) if args.scenario else find_scenario(args.id or trace.get("scenario_id"))
    print(json.dumps(judge(scenario, trace), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
