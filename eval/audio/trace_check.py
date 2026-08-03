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
_ROOT = os.path.dirname(os.path.dirname(_HERE))
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


_SEMANTIC_EVIDENCE_PREFIXES = (
    "card:",
    "contract:",
    "goal:",
    "recipe:",
    "validate:",
    "data:",
    "smoke:",
)
_SEMANTIC_CRITIQUE_SECTIONS = (
    "stage_reviews",
    "field_reviews",
    "behavior_checks",
    "transform_checks",
    "model_checks",
    "assumptions_or_questions",
)


def _semantic_critique(trace: dict) -> dict:
    """Return the structured host-LLM judgement, or an empty mapping.

    ``validate.semantic_review`` is the deterministic evidence packet and is
    normally nested in a tool result. The top-level ``semantic_critique`` is the
    host's distinct intent judgement. The legacy key is accepted only when it
    carries the critique contract, not the deterministic packet.
    """
    for key in ("semantic_critique", "semantic_review"):
        value = trace.get(key)
        if isinstance(value, dict) and (key == "semantic_critique" or "intent_status" in value):
            return value
    return {}


def _tool_calls(trace: dict, *verbs: str) -> list[dict]:
    wanted = set(verbs)
    return [
        call
        for call in trace.get("tool_calls", [])
        if isinstance(call, dict) and call.get("verb") in wanted
    ]


def _artifact_mentions(value: object, locator: str) -> bool:
    try:
        return locator.casefold() in json.dumps(value, default=str).casefold()
    except Exception:  # noqa: BLE001
        return False


def _successful_tool_results(trace: dict, *verbs: str) -> list[object]:
    """Return only inspectable, non-error result payloads for the requested tools.

    Tool arguments prove what the host asked for, not what evidence it received.
    In particular, ``cards(names=["X"])`` must not make ``card:X`` a grounded
    citation when lookup failed or the SDK did not capture the result.
    """
    results: list[object] = []
    for call in _tool_calls(trace, *verbs):
        result = call.get("result")
        if result is None or result == "<result>":
            continue
        if isinstance(result, str):
            continue
        if isinstance(result, dict):
            status = str(result.get("status") or "").casefold()
            if (
                result.get("error")
                or result.get("is_error") is True
                or result.get("isError") is True
                or status in {"error", "failed", "fail", "refused", "blocked"}
            ):
                continue
        results.append(result)
    return results


def _evidence_token_grounded(token: object, scenario: dict, trace: dict) -> bool:
    """Check that a generic ``source:locator`` citation has a trace artifact.

    This verifies provenance of the citation, not truth of the semantic claim.
    The model judge/human rubric remains responsible for meaning.
    """
    if not isinstance(token, str) or ":" not in token:
        return False
    source, locator = token.split(":", 1)
    source = source.casefold().strip()
    locator = locator.strip()
    if not locator or f"{source}:" not in _SEMANTIC_EVIDENCE_PREFIXES:
        return False

    if source == "goal":
        prompt = scenario.get("prompt") or trace.get("prompt") or ""
        return bool(prompt.strip()) and (
            locator in {"user-request", "prompt", "objective"}
            or _artifact_mentions(prompt, locator)
        )
    if source == "recipe":
        recipe = trace.get("final_recipe")
        return bool(recipe) and (
            locator in {"draft", "final", "pipeline"}
            or _artifact_mentions(recipe, locator)
        )
    if source == "card":
        results = _successful_tool_results(trace, "cards", "describe")
        return bool(results) and _artifact_mentions(results, locator)
    if source == "contract":
        artifacts: list[object] = []
        for result in _successful_tool_results(trace, "describe"):
            artifacts.append(result)
        for result in _successful_tool_results(trace, "validate"):
            if isinstance(result, dict) and isinstance(
                result.get("semantic_review"), dict
            ):
                artifacts.append(result["semantic_review"])
        return bool(artifacts) and _artifact_mentions(artifacts, locator)
    if source == "validate":
        result = _latest_tool_result(trace, "validate")
        return result in _successful_tool_results(trace, "validate") and (
            locator
            in {
                "result",
                "runnable",
                "mechanically-runnable",
                "semantic-review",
                "config-hash",
            }
            or _artifact_mentions(result, locator)
        )
    if source == "data":
        results = _successful_tool_results(trace, "context")
        return bool(results) and (
            locator in {"profile", "input", "dataset"}
            or _artifact_mentions(results, locator)
        )
    if source == "smoke":
        results = _successful_tool_results(trace, "smoke")
        return bool(results) and (
            locator in {"result", "sample", "evidence"}
            or _artifact_mentions(results, locator)
        )
    return False


def _latest_tool_result(trace: dict, verb: str) -> object:
    for call in reversed(_tool_calls(trace, verb)):
        if "result" in call:
            return call.get("result")
    return None


def _latest_tool_call(trace: dict, verb: str) -> dict | None:
    calls = _tool_calls(trace, verb)
    return calls[-1] if calls else None


def _ordered_refs(refs: list[str], expected: list[str]) -> tuple[bool, str]:
    """Check an exact stage-id subsequence without any module-specific aliases."""
    cursor = 0
    positions: list[int] = []
    for wanted in expected:
        try:
            pos = refs.index(wanted, cursor)
        except ValueError:
            return False, f"required order token {wanted!r} is missing after position {cursor - 1}"
        positions.append(pos)
        cursor = pos + 1
    return True, f"required stage subsequence present at {positions}"


def _stage_params_match(recipe: dict, expectation: dict) -> tuple[bool, str]:
    """Match explicit params on one occurrence of a stage.

    Semantic regressions intentionally require explicit values where relying on
    a default would obscure the decision under test. This matcher is generic
    over stage ids and parameter names; all domain meaning stays in YAML.
    """
    stage_id = expectation.get("stage")
    occurrence = expectation.get("occurrence", 0)
    if not isinstance(stage_id, str) or not isinstance(occurrence, int) or occurrence < 0:
        return False, f"invalid stage-param expectation {expectation!r}"
    matches = [
        stage for stage in recipe.get("stages", [])
        if isinstance(stage, dict) and stage.get("ref") == stage_id
    ]
    if occurrence >= len(matches):
        return False, f"{stage_id} occurrence {occurrence} is missing"
    actual = matches[occurrence].get("params") or {}
    expected = expectation.get("params") or {}
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False, f"{stage_id} params/expectation is not a mapping"
    wrong = [
        f"{name}={actual.get(name)!r} expected {value!r}"
        for name, value in expected.items()
        if name not in actual or actual.get(name) != value
    ]
    return (not wrong), ("; ".join(wrong) or f"{stage_id} explicit semantic params match")


def _contains_subset(actual: object, expected: object) -> bool:
    """Recursive unordered subset matcher for generic recipe expectations."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(
            any(_contains_subset(candidate, wanted) for candidate in actual)
            for wanted in expected
        )
    return actual == expected


def _live_fixture_path(scenario: dict) -> str | None:
    live = scenario.get("live_capture") or {}
    rel = live.get("fixture") if isinstance(live, dict) else None
    if not isinstance(rel, str) or not rel.strip():
        return None
    path = os.path.realpath(os.path.join(_ROOT, rel))
    root = os.path.realpath(_ROOT)
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        return None
    return path


def _recipe_binds_fixture(recipe: object, fixture: str) -> bool:
    """Require the concrete manifest on a source stage, not merely in tool args."""
    if not isinstance(recipe, dict):
        return False
    for stage in recipe.get("stages", []) or []:
        if not isinstance(stage, dict) or stage.get("ref") != "ManifestReader":
            continue
        params = stage.get("params") or {}
        configured = params.get("manifest_path") if isinstance(params, dict) else None
        if not isinstance(configured, str) or not configured.strip():
            continue
        candidate = os.path.realpath(
            configured
            if os.path.isabs(configured)
            else os.path.join(_ROOT, configured)
        )
        if candidate == fixture:
            return True
    return False


def _grade_semantic_expectations(scenario: dict, trace: dict) -> tuple[bool, list[str], bool]:
    """Grade the generic host critique structure plus its recipe consequence.

    Returns ``(ok, problems, critique_missing)``. The grader never knows what
    ``num_speakers`` or any other domain field means; scenarios state the gold
    sections and recipe consequences, while the host must cite actual trace
    artifacts. A model judge separately evaluates whether the prose is correct.
    """
    spec = scenario.get("semantic_expectations") or {}
    if not isinstance(spec, dict) or not spec:
        return True, [], False

    problems: list[str] = []
    workflow_spec = spec.get("workflow") or {}
    live_fixture = _live_fixture_path(scenario)
    if scenario.get("live_capture") and live_fixture is None:
        problems.append("semantic live-capture fixture is missing or unsafe")
    if live_fixture is not None:
        doctor_result = _latest_tool_result(trace, "doctor")
        if workflow_spec.get("require_doctor", True) and not isinstance(
            doctor_result, dict
        ):
            problems.append(
                "semantic live-capture workflow did not inspect the real host with doctor"
            )
    if workflow_spec.get("require_card_inspection", True) and not _tool_calls(
        trace, "cards", "describe"
    ):
        problems.append("semantic workflow did not inspect cards or stage descriptions")

    validate_call = _latest_tool_call(trace, "validate")
    validate_result = _latest_tool_result(trace, "validate")
    if workflow_spec.get("require_validate", True) and validate_call is None:
        problems.append("semantic workflow did not validate the candidate recipe")
    if workflow_spec.get("require_green_validate", True):
        if not isinstance(validate_result, dict):
            problems.append("semantic workflow has no inspectable validate result")
        elif (
            validate_result.get("runnable") is not True
            or validate_result.get("status") != "pass"
        ):
            problems.append(
                "semantic workflow did not reach status=pass and runnable=true"
            )
    if workflow_spec.get("require_evidence_packet", True):
        if not isinstance(validate_result, dict) or not isinstance(
            validate_result.get("semantic_review"), dict
        ):
            problems.append(
                "validate result did not expose the deterministic semantic_review packet"
            )
    if workflow_spec.get("require_final_recipe_validated", True) and validate_call:
        validated_recipe = (validate_call.get("args") or {}).get("recipe")
        if not isinstance(validated_recipe, dict):
            problems.append("cannot prove that validate checked the final recipe")
        elif validated_recipe != (trace.get("final_recipe") or {}):
            problems.append("the final recipe differs from the recipe last validated")
    if live_fixture is not None:
        if not _recipe_binds_fixture(trace.get("final_recipe"), live_fixture):
            problems.append(
                "final recipe is not bound to the scenario's concrete input fixture"
            )
        if validate_call and not _recipe_binds_fixture(
            (validate_call.get("args") or {}).get("recipe"), live_fixture
        ):
            problems.append(
                "validate did not check a recipe bound to the concrete input fixture"
            )
    if workflow_spec.get("forbid_execution_before_critique", True) and _tool_calls(
        trace, "smoke", "run"
    ):
        problems.append(
            "semantic workflow executed smoke/run before the captured intent critique"
        )

    critique_spec = spec.get("critique") or spec.get("review") or {}
    critique = _semantic_critique(trace)
    critique_missing = not critique
    if critique_spec.get("required", True) and critique_missing:
        problems.append("semantic critique artifact is missing")
    if critique:
        packet_recipe = (
            validate_result.get("semantic_review", {}).get("recipe", {})
            if isinstance(validate_result, dict)
            and isinstance(validate_result.get("semantic_review"), dict)
            else {}
        )
        packet_hash = (
            packet_recipe.get("config_hash")
            if isinstance(packet_recipe, dict)
            else None
        )
        critique_hash = critique.get("recipe_config_hash")
        if not isinstance(packet_hash, str) or not packet_hash:
            problems.append(
                "validate semantic_review packet has no recipe config_hash"
            )
        if not isinstance(critique_hash, str) or not critique_hash:
            problems.append("semantic critique has no recipe_config_hash")
        elif isinstance(packet_hash, str) and critique_hash != packet_hash:
            problems.append(
                "semantic critique recipe_config_hash disagrees with the validated recipe"
            )
        wanted_status = critique_spec.get("intent_status", critique_spec.get("status"))
        if wanted_status is not None and critique.get("intent_status") != wanted_status:
            problems.append(
                f"semantic critique intent_status={critique.get('intent_status')!r} "
                f"expected {wanted_status!r}"
            )
        wanted_runnable = critique_spec.get("mechanically_runnable")
        if (
            wanted_runnable is not None
            and critique.get("mechanically_runnable") is not wanted_runnable
        ):
            problems.append(
                "semantic critique mechanically_runnable="
                f"{critique.get('mechanically_runnable')!r} expected {wanted_runnable!r}"
            )
        if (
            isinstance(validate_result, dict)
            and "runnable" in validate_result
            and critique.get("mechanically_runnable") != validate_result["runnable"]
        ):
            problems.append(
                "semantic critique mechanically_runnable disagrees with the validate result"
            )

        required_sections = (
            critique_spec.get("required_sections") or _SEMANTIC_CRITIQUE_SECTIONS
        )
        sections: dict[str, list] = {}
        for section in required_sections:
            value = critique.get(section)
            if not isinstance(value, list):
                problems.append(f"semantic critique section {section!r} must be a list")
                sections[section] = []
            else:
                sections[section] = value

        minimums = critique_spec.get("min_items_by_section") or {}
        for section, minimum in minimums.items():
            values = sections.get(section)
            if values is None:
                value = critique.get(section)
                values = value if isinstance(value, list) else []
            if len(values) < int(minimum):
                problems.append(
                    f"semantic critique section {section!r} has {len(values)} item(s), "
                    f"expected at least {minimum}"
                )
        for section in critique_spec.get("empty_sections", []) or []:
            value = critique.get(section)
            if isinstance(value, list) and value:
                problems.append(f"semantic critique section {section!r} must be empty")

        if critique_spec.get("cover_all_recipe_stages"):
            reviewed = {
                item.get("stage")
                for item in sections.get("stage_reviews", [])
                if isinstance(item, dict)
            }
            missing_stages = [ref for ref in _recipe_refs(trace) if ref not in reviewed]
            if missing_stages:
                problems.append(
                    f"semantic critique omitted stage review(s) for {missing_stages}"
                )

        evidence_sections = (
            critique_spec.get("evidence_sections")
            or list((critique_spec.get("min_items_by_section") or {}).keys())
        )
        min_finding_chars = int(critique_spec.get("min_finding_chars", 16))
        for section in evidence_sections:
            value = critique.get(section)
            if not isinstance(value, list):
                continue
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    problems.append(
                        f"semantic critique {section}[{index}] must be a mapping"
                    )
                    continue
                finding = item.get("finding")
                if (
                    not isinstance(finding, str)
                    or len(finding.strip()) < min_finding_chars
                ):
                    problems.append(
                        f"semantic critique {section}[{index}] has no substantive finding"
                    )
                evidence = item.get("evidence")
                if not isinstance(evidence, list) or not evidence:
                    problems.append(
                        f"semantic critique {section}[{index}] has no evidence citations"
                    )
                else:
                    ungrounded = [
                        token
                        for token in evidence
                        if not _evidence_token_grounded(token, scenario, trace)
                    ]
                    if ungrounded:
                        problems.append(
                            f"semantic critique {section}[{index}] has ungrounded "
                            f"evidence {ungrounded!r}"
                        )

    recipe_spec = spec.get("recipe") or {}
    recipe = trace.get("final_recipe") or {}
    refs = _recipe_refs(trace)
    required_order = recipe_spec.get("required_order") or []
    if required_order:
        order_ok, note = _ordered_refs(refs, required_order)
        if not order_ok:
            problems.append(note)
    for stage_id in recipe_spec.get("absent_stages", []) or []:
        if stage_id in refs:
            problems.append(f"stage {stage_id!r} must be absent for this intent")
    for stage_id in recipe_spec.get("required_stages", []) or []:
        if stage_id not in refs:
            problems.append(f"stage {stage_id!r} is required for this intent")
    for expectation in recipe_spec.get("stage_params", []) or []:
        if not isinstance(expectation, dict):
            problems.append(f"invalid stage-param expectation {expectation!r}")
            continue
        params_ok, note = _stage_params_match(recipe, expectation)
        if not params_ok:
            problems.append(note)
    contains = recipe_spec.get("contains")
    if contains is not None and not _contains_subset(recipe, contains):
        problems.append("final recipe is missing required semantic intent fields")

    return not problems, problems, critique_missing


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

    # 8) Semantic critic: generic structured host judgement + intent-specific recipe
    # consequence. Mechanical validation can pass for every counterexample in
    # this suite, so this is intentionally a separate hard dimension.
    if scenario.get("semantic_expectations"):
        sem_ok, sem_problems, critique_missing = _grade_semantic_expectations(
            scenario, trace
        )
        dims["semantic_critique"] = {
            "pass": sem_ok,
            "note": "; ".join(sem_problems)
            or "grounded semantic critique and recipe consequence match",
        }
        if not sem_ok:
            add(
                "semantic_critique_missing"
                if critique_missing
                else "intent_misunderstanding"
            )

    # 9) Continuation (Efficient) — only when the scenario expects a reuse plan
    cont_exp = scenario.get("continuation_expect")
    if cont_exp:
        got = _continuation(trace)
        mode_ok = got.get("mode") == cont_exp.get("mode")
        dims["efficient"] = {"pass": mode_ok, "note": f"continuation mode={got.get('mode')} exp={cont_exp.get('mode')}"}
        if not mode_ok:
            add("missed_reuse")

    # 10) Safe — agent-trace safety signals derived from the captured tool calls
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
    "scenario_id": "L01_duration",
    "tool_calls": [
        {"verb": "context", "args": {"goal": {"task": "duration"}}},
        {"verb": "cards", "args": {"names": ["GetAudioDurationStage"]}},
        {"verb": "validate", "args": {}, "result": {"runnable": True, "status": "pass"}},
    ],
    "final_recipe": {"stages": [
        {
            "ref": "ManifestReader",
            "params": {
                "manifest_path": os.path.join(
                    _ROOT, "tests", "fixtures", "audio", "alm", "sample_input.jsonl"
                )
            },
        },
        {"ref": "GetAudioDurationStage", "params": {}},
    ]},
    "refused": False,
    "explanation": (
        "Duration is computed independently for each clip. No filtering or "
        "transcription was added because neither was requested."
    ),
}

_BAD_TRACE = {
    "scenario_id": "L01_duration",
    "tool_calls": [
        {"verb": "validate", "args": {}, "result": {"runnable": True}},
    ],
    "final_recipe": {"stages": [
        {
            "ref": "ManifestReader",
            "params": {
                "manifest_path": os.path.join(
                    _ROOT, "tests", "fixtures", "audio", "alm", "sample_input.jsonl"
                )
            },
        },
        {"ref": "MonoConversionStage", "params": {}},
    ]},
    "refused": False,
    "explanation": "",
}


def _selftest() -> int:
    scenario = find_scenario("L01_duration")
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
