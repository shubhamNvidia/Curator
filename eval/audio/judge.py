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
    python -m eval.audio.judge --id S15_segment_quality --trace traces/S15.json --model claude-opus-4-8
    python -m eval.audio.judge --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

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


def _semantic_critique_items(scenario: dict, trace: dict) -> tuple[list[dict], bool]:
    """Score generic host-critique coverage/citations, never domain meaning."""
    from eval.audio.trace_check import (
        _evidence_token_grounded,
        _semantic_critique,
    )

    semantic_spec = scenario.get("semantic_expectations") or {}
    spec = semantic_spec.get("critique") or semantic_spec.get("review") or {}
    if not spec:
        return [], True
    critique = _semantic_critique(trace)

    items: list[dict] = []
    grounded = True
    minimums = spec.get("min_items_by_section") or {}
    for section, minimum in minimums.items():
        values = critique.get(section) if isinstance(critique, dict) else None
        values = values if isinstance(values, list) else []
        cited = [
            item
            for item in values
            if isinstance(item, dict)
            and isinstance(item.get("finding"), str)
            and len(item["finding"].strip()) >= int(spec.get("min_finding_chars", 16))
            and isinstance(item.get("evidence"), list)
            and item["evidence"]
            and all(
                _evidence_token_grounded(token, scenario, trace)
                for token in item["evidence"]
            )
        ]
        met = len(cited) >= int(minimum)
        grounded = grounded and met
        items.append(
            {
                "criterion": (
                    f"semantic critique has at least {minimum} substantive grounded "
                    f"item(s) in {section}"
                ),
                "score": 1.0 if met else 0.0,
                "met": met,
            }
        )
    wanted_status = spec.get("intent_status", spec.get("status"))
    if wanted_status is not None:
        met = bool(critique) and critique.get("intent_status") == wanted_status
        items.append(
            {
                "criterion": f"semantic critique intent_status is {wanted_status!r}",
                "score": 1.0 if met else 0.0,
                "met": met,
            }
        )
        grounded = grounded and met
    wanted_runnable = spec.get("mechanically_runnable")
    if wanted_runnable is not None:
        met = bool(critique) and critique.get("mechanically_runnable") is wanted_runnable
        items.append(
            {
                "criterion": (
                    "semantic critique copies mechanically_runnable="
                    f"{wanted_runnable!r}"
                ),
                "score": 1.0 if met else 0.0,
                "met": met,
            }
        )
        grounded = grounded and met
    return items, grounded


def _deterministic(scenario: dict, trace: dict) -> dict:
    rubric = scenario.get("explanation_rubric", []) or []
    expl = (trace.get("explanation") or "").strip()
    expl_kw = _keywords(expl)
    items = []
    for r in rubric:
        kw = _keywords(r)
        score = 1.0 if not kw else round(len(kw & expl_kw) / len(kw), 3)
        items.append({"criterion": r, "score": score, "met": score >= _PASS_ITEM})
    semantic_items, semantic_grounded = _semantic_critique_items(scenario, trace)
    items.extend(semantic_items)
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
        "grounded": not ungrounded and semantic_grounded,
        "ungrounded_numbers": ungrounded,
        "note": "keyword-grounding fallback (indicative only, not a semantic quality score); pass model= for real judgement",
    }


def _validated_model_response(data: object) -> dict:
    """Validate the authoritative judge contract without JSON truthiness coercions."""
    if not isinstance(data, dict):
        raise ValueError("judge response must be a JSON object")

    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("judge response items must be a non-empty list")
    normalized_items: list[dict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"judge item {index} must be an object")
        criterion = item.get("criterion")
        score = item.get("score")
        met = item.get("met")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"judge item {index} criterion must be a non-empty string")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"judge item {index} score must be a finite number in [0, 1]")
        if not isinstance(met, bool):
            raise ValueError(f"judge item {index} met must be a boolean")
        normalized_items.append(
            {"criterion": criterion.strip(), "score": float(score), "met": met}
        )

    overall = data.get("overall")
    if (
        isinstance(overall, bool)
        or not isinstance(overall, (int, float))
        or not math.isfinite(float(overall))
        or not 0.0 <= float(overall) <= 1.0
    ):
        raise ValueError("judge overall must be a finite number in [0, 1]")
    grounded = data.get("grounded")
    if not isinstance(grounded, bool):
        raise ValueError("judge grounded must be a boolean")
    notes = data.get("notes")
    if not isinstance(notes, str):
        raise ValueError("judge notes must be a string")
    return {
        "items": normalized_items,
        "overall": float(overall),
        "grounded": grounded,
        "notes": notes,
    }


def _model_judge(scenario: dict, trace: dict, model) -> dict:  # noqa: ANN001
    prompt = (
        "You are grading an audio-curation agent's explanation and semantic critique against a rubric. "
        "Mechanical validation is necessary but does not prove that field provenance, granularity, "
        "metric scope, transform use, or explicit key selection matches the user's intent. "
        "Return STRICT JSON: {\"items\":[{\"criterion\":str,\"score\":0..1,\"met\":bool}],"
        "\"overall\":0..1,\"grounded\":bool,\"notes\":str}.\n\n"
        f"PROMPT: {scenario.get('prompt')}\n"
        f"RUBRIC: {json.dumps(scenario.get('explanation_rubric', []))}\n"
        f"SEMANTIC EXPECTATIONS: {json.dumps(scenario.get('semantic_expectations', {}))}\n"
        f"MECHANICALLY VALID COUNTEREXAMPLE: {json.dumps((scenario.get('semantic_pair') or {}).get('mechanically_valid_counterexample', {}))}\n"
        f"AGENT SEMANTIC CRITIQUE: {json.dumps(trace.get('semantic_critique') or {})}\n"
        f"FINAL RECIPE: {json.dumps(trace.get('final_recipe') or {})}\n"
        f"AGENT EXPLANATION: {trace.get('explanation', '')}\n"
        f"OBSERVED TOOL RESULTS: {_trace_result_text(trace)}\n"
    )
    raw = model(prompt)
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - allow a fenced object, then validate strictly
        m = re.search(r"\{.*\}", str(raw), re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else None
        except Exception:  # noqa: BLE001
            data = None
    try:
        result = _validated_model_response(data)
    except ValueError as exc:
        return {
            "mode": "model",
            "authoritative": False,
            "scenario_id": scenario.get("id"),
            "items": [],
            "overall": 0.0,
            "grounded": False,
            "notes": "",
            "validation_error": str(exc),
        }
    result["mode"] = "model"
    result["authoritative"] = True
    result["scenario_id"] = scenario.get("id")
    return result


def judge(scenario: dict, trace: dict, *, model=None) -> dict:  # noqa: ANN001
    """Score the qualitative rubric. Uses ``model`` if given, else the fallback."""
    return _model_judge(scenario, trace, model) if model else _deterministic(scenario, trace)


def cursor_model(model_name: str, api_key: str | None = None):  # noqa: ANN201
    """Return a simple Cursor SDK callable for authoritative model judging."""
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

    options = AgentOptions(
        model=model_name,
        api_key=api_key or os.environ.get("CURSOR_API_KEY"),
        local=LocalAgentOptions(cwd=_ROOT),
    )

    def _call(prompt: str) -> str:
        response = Agent.prompt(prompt, options)
        return str(getattr(response, "result", "") or "")

    return _call


def _selftest() -> int:
    from eval.audio.trace_check import _BAD_TRACE, _GOOD_TRACE, find_scenario

    scenario = find_scenario("L01_duration")
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
    ap.add_argument("--model", help="Cursor model id for authoritative semantic judgement")
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
    model = cursor_model(args.model) if args.model else None
    print(json.dumps(judge(scenario, trace, model=model), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
