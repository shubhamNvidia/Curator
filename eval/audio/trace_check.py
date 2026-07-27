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

"""Grade a captured LLM-plane agent trace against a scenario (AGENT_TEST_PLAN.md 6.1/7.1).

A *trace* is what the host agent actually did for a scenario prompt:

    {
      "scenario_id": "L02_readspeech_quality",
      "tool_calls": [
        {"verb": "context", "args": {...}},
        {"clarification": "studio, general, or lenient?"},
        {"verb": "resolve", "args": {"stage": "UTMOSFilterStage", "label": "studio"}},
        {"verb": "validate", "args": {...}, "result": {"runnable": true, "status": "pass"}}
      ],
      "final_recipe": {"stages": [{"ref": "...", "params": {...}}, ...]},
      "refused": false,
      "explanation": "…the agent's user-facing rationale…"
    }

Grading is deterministic and decoupled from capture: it scores the 7 north-star
dimensions (Appropriate/Complete/Compatible/Configured/Efficient/Recoverable/
Explainable) plus clarification and refusal, re-validates the final recipe with
the real core, and maps every miss to a taxonomy signal.

    python -m eval.audio.trace_check --scenario eval/audio/scenarios/L02_standard_multistage.yaml \
        --id L02_readspeech_quality --trace eval/audio/traces/L02_readspeech_quality.json
    python -m eval.audio.trace_check --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TAXONOMY = os.path.join(_HERE, "taxonomy.yaml")
_SCENARIOS_DIR = os.path.join(_HERE, "scenarios")
_TRACES_DIR = os.path.join(_HERE, "traces")


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenario(path: str, sid: str | None = None) -> dict:
    """Load one scenario from a scenarios file (by id, else the first)."""
    doc = _load_yaml(path)
    scs = doc.get("scenarios", [])
    if sid:
        for s in scs:
            if s.get("id") == sid:
                return s
        msg = f"scenario id {sid!r} not found in {path}"
        raise KeyError(msg)
    if not scs:
        msg = f"no scenarios in {path}"
        raise ValueError(msg)
    return scs[0]


def find_scenario(sid: str) -> dict:
    """Search all scenarios/*.yaml for a scenario id."""
    import glob

    for f in sorted(glob.glob(os.path.join(_SCENARIOS_DIR, "*.yaml"))):
        for s in _load_yaml(f).get("scenarios", []):
            if s.get("id") == sid:
                return s
    msg = f"scenario id {sid!r} not found under {_SCENARIOS_DIR}"
    raise KeyError(msg)


# --------------------------------------------------------------------------- #
# Trace inspection helpers
# --------------------------------------------------------------------------- #

def _verbs(trace: dict) -> list[str]:
    return [tc.get("verb") for tc in trace.get("tool_calls", []) if tc.get("verb")]


def _asked_clarification(trace: dict) -> bool:
    for tc in trace.get("tool_calls", []):
        if tc.get("clarification") or tc.get("verb") in ("ask", "clarify"):
            return True
    return bool(trace.get("clarification"))


def _recipe_refs(trace: dict) -> list[str]:
    rec = trace.get("final_recipe") or {}
    return [s.get("ref") for s in rec.get("stages", []) if s.get("ref")]


_TRANSCRIPT_KEYS = ("text", "pred_text", "reference_text", "transcript", "text_ref")


def _continuation(trace: dict) -> dict:
    """The continuation plan: the explicit trace field, else the plan_continuation tool result.

    agent_runner records verbs+results but leaves the top-level ``continuation`` field
    None, so deriving it from the captured tool call keeps the Efficient dimension
    satisfiable for real traces (previously it always missed).
    """
    if isinstance(trace.get("continuation"), dict) and trace["continuation"]:
        return trace["continuation"]
    for tc in trace.get("tool_calls", []):
        if tc.get("verb") == "plan_continuation" and isinstance(tc.get("result"), dict):
            return tc["result"]
    return {}


def _transcript_leak_count(trace: dict) -> int:
    """Count tool-result values under a transcript key that are NOT redaction-masked.

    Redaction replaces transcript text with ``<redacted-transcript:...>``; a raw string
    under such a key means transcripts/secrets reached the host (a redaction regression).
    """

    def _walk(obj: object) -> int:
        n = 0
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in _TRANSCRIPT_KEYS and isinstance(v, str) and v and not v.startswith("<redacted"):
                    n += 1
                else:
                    n += _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                n += _walk(v)
        return n

    return sum(_walk(tc.get("result")) for tc in trace.get("tool_calls", []))


def _is_refused(trace: dict) -> bool:
    if trace.get("refused"):
        return True
    rec = trace.get("final_recipe")
    return not rec or not rec.get("stages")


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def _grade_modules(scenario: dict, refs: list[str]) -> tuple[bool, list[str]]:
    spec = scenario.get("expected_modules", {}) or {}
    problems: list[str] = []
    refset = set(refs)
    for req in spec.get("required", []) or []:
        if req not in refset:
            problems.append(f"missing required module {req}")
    for forb in spec.get("forbidden", []) or []:
        if forb in refset:
            problems.append(f"forbidden module present {forb}")
    for group, members in (spec.get("equivalence", {}) or {}).items():
        if not (set(members) & refset):
            problems.append(f"no member of equivalence group {group!r} present ({members})")
    return (not problems), problems


def _grade_validation(scenario: dict, trace: dict) -> tuple[bool | None, str, dict]:
    """Re-validate the final recipe with the real core; None if not applicable."""
    if scenario.get("refuse"):
        return None, "refusal scenario -> no recipe to validate", {}
    rec = trace.get("final_recipe")
    if not rec or not rec.get("stages"):
        return False, "no final recipe to validate", {}
    from nemo_curator import audio_agent as aa

    v = aa.validate(rec, expected_outputs=scenario.get("expected_output_roles") or None)
    want = scenario.get("expected_validation", {}) or {}
    ok = True
    bits = []
    if "runnable" in want and v.get("runnable") != want["runnable"]:
        ok = False
        bits.append(f"runnable={v.get('runnable')} exp={want['runnable']}")
    if "status" in want and v.get("status") != want["status"]:
        ok = False
        bits.append(f"status={v.get('status')} exp={want['status']}")
    return ok, ("; ".join(bits) or f"validate status={v.get('status')} runnable={v.get('runnable')}"), v


def grade(scenario: dict, trace: dict) -> dict:  # noqa: C901 - one linear scorecard
    """Return a scorecard {overall, dimensions{...}, violations[...]}. """
    refs = _recipe_refs(trace)
    dims: dict[str, dict] = {}
    violations: list[dict] = []

    def add(sig: str) -> None:
        violations.append(sig)

    # 1) Clarification (Recoverable/clarify)
    required = bool(scenario.get("clarification_required") or scenario.get("expect_clarification"))
    asked = _asked_clarification(trace)
    if required and not asked:
        dims["clarification"] = {"pass": False, "note": "should have asked but did not"}
        add("missing_clarification")
    elif (not required) and asked and not scenario.get("refuse"):
        dims["clarification"] = {"pass": True, "note": "asked though not required (over-clarification)", "warn": True}
        add("over_clarification")
    else:
        dims["clarification"] = {"pass": True, "note": "clarification behavior as expected"}

    # 2) Refusal / deferral
    refuse_expected = bool(scenario.get("refuse"))
    refused = _is_refused(trace)
    deferred = required and asked and not refuse_expected  # correctly asked on an ambiguous prompt
    if refuse_expected:
        dims["refusal"] = {"pass": refused, "note": "refused as expected" if refused else "should have refused/redirected"}
        if not refused:
            add("wrong_module")
    elif deferred:
        dims["refusal"] = {"pass": True, "note": "correctly asked a clarification instead of guessing"}
    else:
        dims["refusal"] = {"pass": (not refused), "note": "produced a pipeline" if not refused else "gave up without asking or building"}
        if refused:
            add("failed_recovery")

    # A recipe is only expected when the agent should neither refuse nor (in a
    # one-shot) correctly defer on an ambiguous prompt by asking a clarification.
    has_recipe = bool((trace.get("final_recipe") or {}).get("stages"))
    # Grade a recipe whenever one is present; only waive it when the agent correctly
    # deferred (asked on an ambiguous prompt) or refused and produced none.
    recipe_expected = not refuse_expected and (not deferred or has_recipe)
    na = "n/a (agent correctly asked/refused; no recipe expected in a one-shot)"

    # 3) Modules (Appropriate + Complete)
    if recipe_expected:
        mod_ok, mod_problems = _grade_modules(scenario, refs)
        dims["modules"] = {"pass": mod_ok, "note": "; ".join(mod_problems) or "module set matches expectations"}
        if not mod_ok:
            add("wrong_module")
    else:
        # even when deferring/refusing, must not fabricate a forbidden stage
        forb = set(scenario.get("expected_modules", {}).get("forbidden", []) or [])
        bad = forb & set(refs)
        dims["modules"] = {"pass": not bad, "note": f"fabricated {sorted(bad)}" if bad else na, "soft": not bad}
        if bad:
            add("wrong_module")

    # 4) Configured (thresholds resolved, not hand-picked)
    must_resolve = (scenario.get("expected_params", {}) or {}).get("must_resolve", []) or []
    if must_resolve and recipe_expected:
        resolved = "resolve" in _verbs(trace)
        dims["configured"] = {"pass": resolved, "note": f"resolve used for {must_resolve}" if resolved else f"{must_resolve} set without resolve (hand-picked)"}
        if not resolved:
            add("hand_picked_threshold")
    else:
        dims["configured"] = {"pass": True, "note": "no thresholds require resolution"}

    # 5) Compatible (final recipe re-validates) — only when a recipe is expected
    verdict: dict = {}
    if recipe_expected:
        val_ok, val_note, verdict = _grade_validation(scenario, trace)
        dims["compatible"] = {"pass": (val_ok is not False), "note": val_note}
        if val_ok is False:
            add("wrong_module")
    else:
        dims["compatible"] = {"pass": True, "note": na, "soft": True}

    # 6) Complete (expected output roles produced)
    want_roles = scenario.get("expected_output_roles") or []
    if want_roles and verdict and recipe_expected:
        produced = set(verdict.get("produced_roles", []))
        missing = [r for r in want_roles if r not in produced]
        dims["completeness"] = {"pass": not missing, "note": f"missing roles {missing}" if missing else "all output roles produced"}
        if missing:
            add("wrong_capability")
    else:
        dims["completeness"] = {"pass": True, "note": "no output-role expectation (or not validatable)"}

    # 7) Explainable (soft — presence + non-empty; deep grading in judge.py)
    expl = (trace.get("explanation") or "").strip()
    if scenario.get("explanation_rubric"):
        dims["explainable"] = {"pass": bool(expl), "note": "explanation present" if expl else "no explanation captured", "soft": True}
        if not expl:
            add("poor_explanation")
    else:
        dims["explainable"] = {"pass": True, "note": "no rubric"}

    # 8) Continuation (Efficient) — only when the scenario expects a reuse plan
    cont_exp = scenario.get("continuation_expect")
    if cont_exp:
        got = _continuation(trace)
        mode_ok = got.get("mode") == cont_exp.get("mode")
        dims["efficient"] = {"pass": mode_ok, "note": f"continuation mode={got.get('mode')} exp={cont_exp.get('mode')}"}
        if not mode_ok:
            add("missed_reuse")

    # 9) Safe — agent-trace safety signals derived from the captured tool calls
    run_call = next((tc for tc in trace.get("tool_calls", []) if tc.get("verb") == "run"), None)
    if run_call is not None:
        confirmed = bool((run_call.get("args") or {}).get("confirm"))
        dims["safe_confirm"] = {
            "pass": confirmed,
            "note": "run() confirmed" if confirmed else "run() called without confirm (confirm-gate bypassed)",
        }
        if not confirmed:
            add("skipped_confirm")
    leaks = _transcript_leak_count(trace)
    dims["safe_redaction"] = {
        "pass": leaks == 0,
        "note": "no transcript leakage in tool results" if leaks == 0 else f"{leaks} raw transcript value(s) leaked",
    }
    if leaks:
        add("transcript_leak")

    # hard dimensions gate overall; soft ones (explainable/warn) do not
    hard = [k for k, d in dims.items() if not d.get("soft") and not d.get("warn")]
    overall = all(dims[k]["pass"] for k in hard)
    return {"scenario_id": scenario.get("id"), "overall": "pass" if overall else "fail",
            "dimensions": dims, "violations": violations}


def classify_violations(violations: list[str]) -> list[dict]:
    """Map LLM-plane signals to taxonomy category/severity."""
    try:
        tax = _load_yaml(_TAXONOMY).get("llm_signals", {})
    except Exception:  # noqa: BLE001
        tax = {}
    out = []
    for sig in violations:
        meta = tax.get(sig, {})
        out.append({"signal": sig, "category": meta.get("category"), "severity": meta.get("severity")})
    return out


# --------------------------------------------------------------------------- #
# Self-test (bundled example traces prove the grader end-to-end)
# --------------------------------------------------------------------------- #

_GOOD_TRACE = {
    "scenario_id": "L02_readspeech_quality",
    "tool_calls": [
        {"verb": "context", "args": {"goal": {"task": "quality_filter", "domain": "read"}}},
        {"verb": "cards", "args": {"category": "quality"}},
        {"clarification": "which quality level: studio, general, or lenient?"},
        {"verb": "resolve", "args": {"stage": "UTMOSFilterStage", "label": "studio"}, "result": {"params": {"mos_threshold": 4.0}}},
        {"verb": "validate", "args": {}, "result": {"runnable": True, "status": "pass"}},
    ],
    "final_recipe": {"stages": [
        {"ref": "ManifestReader", "params": {"manifest_path": "REQUIRED"}},
        {"ref": "MonoConversionStage", "params": {}},
        {"ref": "UTMOSFilterStage", "params": {"mos_threshold": 4.0}},
        {"ref": "AudioToDocumentStage", "params": {}},
    ]},
    "refused": False,
    "explanation": "UTMOS is the quality metric; it captures perceived naturalness, not noise specifically. The 4.0 threshold came from the named 'studio' outcome, not hand-picked. No ASR stage is present because there are no transcripts to score against.",
}

_BAD_TRACE = {
    "scenario_id": "L02_readspeech_quality",
    "tool_calls": [
        {"verb": "validate", "args": {}, "result": {"runnable": True}},
    ],
    "final_recipe": {"stages": [
        {"ref": "ManifestReader", "params": {"manifest_path": "REQUIRED"}},
        {"ref": "UTMOSFilterStage", "params": {"mos_threshold": 3.7}},   # hand-picked, no resolve
        {"ref": "InferenceAsrNemoStage", "params": {"model_name": "x"}},  # forbidden (no transcripts)
        {"ref": "AudioToDocumentStage", "params": {}},
    ]},
    "refused": False,
    "explanation": "",
}


def _selftest() -> int:
    scenario = find_scenario("L02_readspeech_quality")
    good = grade(scenario, _GOOD_TRACE)
    bad = grade(scenario, _BAD_TRACE)
    print("[selftest] good trace:", good["overall"], "| bad trace:", bad["overall"])
    ok = good["overall"] == "pass" and bad["overall"] == "fail"
    if not ok:
        print("  good dims:", json.dumps(good["dimensions"], indent=2))
        print("  bad dims:", json.dumps(bad["dimensions"], indent=2))
        print("  bad violations:", classify_violations(bad["violations"]))
    print("[selftest]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade an agent trace against a scenario")
    ap.add_argument("--scenario", help="path to a scenarios/*.yaml file")
    ap.add_argument("--id", help="scenario id (else the first in the file, or resolved from the trace)")
    ap.add_argument("--trace", help="path to a captured trace JSON")
    ap.add_argument("--selftest", action="store_true", help="run bundled example traces")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.trace:
        ap.error("--trace is required (or use --selftest)")
    with open(args.trace, encoding="utf-8") as f:
        trace = json.load(f)

    if args.scenario:
        scenario = load_scenario(args.scenario, args.id)
    else:
        scenario = find_scenario(args.id or trace.get("scenario_id"))

    card = grade(scenario, trace)
    card["violations"] = classify_violations(card["violations"])
    print(json.dumps(card, indent=2))
    return 0 if card["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
