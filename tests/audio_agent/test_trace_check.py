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

"""Unit tests for the LLM-plane trace grader and semantic-intent regressions."""

from copy import deepcopy
from pathlib import Path

import yaml

from eval.audio.agent_runner import (
    _SEMANTIC_DEFAULT,
    _parse_semantic_critique,
    _scenario_request,
)
from eval.audio.aggregate_traces import _semantic_gate
from eval.audio.judge import _semantic_critique_items, judge
from eval.audio.trace_check import (
    _continuation,
    _evidence_token_grounded,
    _grade_semantic_expectations,
    _grade_validation,
    _transcript_leak_count,
)

_ROOT = Path(__file__).resolve().parents[2]
_SEMANTIC_SCENARIOS = _ROOT / "eval" / "audio" / "scenarios" / "L15_semantic_intent.yaml"


def _critique_trace(scenario: dict, recipe: dict) -> dict:
    critique_spec = scenario["semantic_expectations"]["critique"]
    refs = [stage["ref"] for stage in recipe["stages"]]

    def items(section: str) -> list[dict]:
        minimum = int((critique_spec.get("min_items_by_section") or {}).get(section, 0))
        values = []
        for index in range(minimum):
            stage = refs[min(index, len(refs) - 1)]
            item = {
                "finding": (
                    f"The {section} decision is grounded in the requested intent "
                    "and configured data flow."
                ),
                "evidence": [
                    "goal:user-request",
                    f"card:{stage}",
                    f"recipe:{stage}",
                ],
            }
            if section == "stage_reviews":
                item["stage"] = stage
            elif section == "field_reviews":
                item.update({"field": "semantic_field", "producer": refs[0]})
                item["evidence"].append(f"contract:{refs[0]}")
            values.append(item)
        return values

    critique = {
        "mechanically_runnable": True,
        "recipe_config_hash": "test-config-hash",
        "intent_status": "pass",
        "stage_reviews": [
            {
                "stage": stage,
                "finding": (
                    "This configured stage has an explicit purpose in the user request."
                ),
                "evidence": [
                    "goal:user-request",
                    f"card:{stage}",
                    f"recipe:{stage}",
                ],
            }
            for stage in refs
        ],
        "field_reviews": items("field_reviews"),
        "behavior_checks": items("behavior_checks"),
        "transform_checks": items("transform_checks"),
        "model_checks": items("model_checks"),
        "assumptions_or_questions": [],
    }
    return {
        "prompt": scenario["prompt"],
        "tool_calls": [
            {
                "verb": "doctor",
                "args": {},
                "result": {"status": "ok", "checks": []},
            },
            {
                "verb": "cards",
                "args": {"names": refs},
                "result": {"cards": [{"stage_id": ref} for ref in refs]},
            },
            {
                "verb": "validate",
                "args": {"recipe": deepcopy(recipe)},
                "result": {
                    "runnable": True,
                    "status": "pass",
                    "semantic_review": {
                        "status": "complete",
                        "recipe": {"config_hash": "test-config-hash"},
                        "stages": [
                            {"stage_id": ref, "contract": {"available": True}}
                            for ref in refs
                        ],
                    },
                },
            },
        ],
        "semantic_critique": critique,
        "final_recipe": deepcopy(recipe),
        "explanation": "The candidate was reviewed against its intended semantic boundary.",
    }


def _counterexample_recipe(
    counterexample: dict,
    scenario: dict | None = None,
) -> dict:
    stages = deepcopy(counterexample["stages"])
    if scenario is not None:
        fixture = _ROOT / scenario["live_capture"]["fixture"]
        stages.insert(
            0,
            {
                "ref": "ManifestReader",
                "params": {"manifest_path": str(fixture.resolve())},
            },
        )
    recipe = {"stages": stages}
    if "acceptance_criteria" in counterexample:
        recipe["acceptance_criteria"] = deepcopy(counterexample["acceptance_criteria"])
    return recipe


class TestContinuationDerivation:
    def test_derives_from_plan_continuation_tool_result(self) -> None:
        # agent_runner leaves trace['continuation']=None; derive it from the tool call.
        trace = {"tool_calls": [{"verb": "plan_continuation", "result": {"mode": "extend", "run_stages": ["X"]}}]}
        assert _continuation(trace).get("mode") == "extend"

    def test_prefers_explicit_field(self) -> None:
        trace = {
            "continuation": {"mode": "full_rerun"},
            "tool_calls": [{"verb": "plan_continuation", "result": {"mode": "extend"}}],
        }
        assert _continuation(trace).get("mode") == "full_rerun"

    def test_empty_when_absent(self) -> None:
        assert _continuation({"tool_calls": [{"verb": "validate"}]}) == {}


class TestTranscriptLeakDetection:
    def test_flags_raw_transcript(self) -> None:
        leaky = {"tool_calls": [{"verb": "smoke", "result": {"examples": [{"text": "hello world this is a raw transcript"}]}}]}
        assert _transcript_leak_count(leaky) == 1

    def test_ignores_redacted_transcript(self) -> None:
        clean = {"tool_calls": [{"verb": "smoke", "result": {"examples": [{"text": "<redacted-transcript:20chars>"}]}}]}
        assert _transcript_leak_count(clean) == 0

    def test_zero_when_no_transcript_keys(self) -> None:
        assert _transcript_leak_count({"tool_calls": [{"verb": "validate", "result": {"status": "pass"}}]}) == 0


class TestSemanticReviewContract:
    def test_parses_fenced_semantic_critique(self) -> None:
        text = """
```semantic-critique
mechanically_runnable: true
intent_status: pass
stage_reviews:
  - stage: ProducerStage
    finding: The producer serves the requested field operation.
    evidence: [goal:user-request, card:ProducerStage]
field_reviews: []
behavior_checks: []
transform_checks: []
model_checks: []
assumptions_or_questions: []
```
"""
        critique = _parse_semantic_critique(text)
        assert critique is not None
        assert critique["intent_status"] == "pass"
        assert critique["stage_reviews"][0]["stage"] == "ProducerStage"

    def test_generic_grader_checks_review_and_recipe_consequence(self) -> None:
        scenario = {
            "prompt": "Use the producer value in the consumer.",
            "semantic_expectations": {
                "critique": {
                    "required": True,
                    "mechanically_runnable": True,
                    "intent_status": "pass",
                    "required_sections": [
                        "stage_reviews",
                        "field_reviews",
                        "behavior_checks",
                        "transform_checks",
                        "model_checks",
                        "assumptions_or_questions",
                    ],
                    "cover_all_recipe_stages": True,
                    "min_items_by_section": {
                        "stage_reviews": 2,
                        "field_reviews": 1,
                    },
                    "evidence_sections": ["stage_reviews", "field_reviews"],
                },
                "recipe": {
                    "required_order": ["ProducerStage", "ConsumerStage"],
                    "stage_params": [
                        {
                            "stage": "ConsumerStage",
                            "params": {"input_key": "producer_value"},
                        }
                    ],
                },
            }
        }
        recipe = {
            "stages": [
                {"ref": "ProducerStage", "params": {}},
                {"ref": "ConsumerStage", "params": {"input_key": "producer_value"}},
            ]
        }
        trace = _critique_trace(scenario, recipe)
        ok, problems, missing = _grade_semantic_expectations(scenario, trace)
        assert ok
        assert not problems
        assert not missing

        trace["final_recipe"]["stages"][1]["params"]["input_key"] = "other_value"
        ok, problems, missing = _grade_semantic_expectations(scenario, trace)
        assert not ok
        assert not missing
        assert any("other_value" in problem for problem in problems)

    def test_requires_real_workflow_and_resolvable_evidence(self) -> None:
        scenario = {
            "prompt": "Use ProducerStage.",
            "semantic_expectations": {
                "critique": {
                    "intent_status": "pass",
                    "mechanically_runnable": True,
                    "min_items_by_section": {"stage_reviews": 1},
                    "evidence_sections": ["stage_reviews"],
                },
                "recipe": {"required_stages": ["ProducerStage"]},
            },
        }
        recipe = {"stages": [{"ref": "ProducerStage", "params": {}}]}
        trace = _critique_trace(scenario, recipe)

        trace["semantic_critique"]["stage_reviews"][0]["evidence"] = [
            "card:DOES_NOT_EXIST"
        ]
        ok, problems, _ = _grade_semantic_expectations(scenario, trace)
        assert not ok
        assert any("ungrounded evidence" in problem for problem in problems)

        trace = _critique_trace(scenario, recipe)
        trace["tool_calls"] = []
        ok, problems, _ = _grade_semantic_expectations(scenario, trace)
        assert not ok
        assert any("did not validate" in problem for problem in problems)
        assert any("did not inspect cards" in problem for problem in problems)

        trace = _critique_trace(scenario, recipe)
        trace["tool_calls"][-1]["result"].update(
            {"runnable": False, "status": "fail"}
        )
        ok, problems, _ = _grade_semantic_expectations(scenario, trace)
        assert not ok
        assert any("runnable=true" in problem for problem in problems)
        assert any("disagrees with the validate result" in problem for problem in problems)

        trace = _critique_trace(scenario, recipe)
        del trace["semantic_critique"]["model_checks"]
        ok, problems, _ = _grade_semantic_expectations(scenario, trace)
        assert not ok
        assert any("'model_checks' must be a list" in problem for problem in problems)

        trace = _critique_trace(scenario, recipe)
        trace["semantic_critique"]["recipe_config_hash"] = "different-recipe"
        ok, problems, _ = _grade_semantic_expectations(scenario, trace)
        assert not ok
        assert any("recipe_config_hash disagrees" in problem for problem in problems)

    def test_card_and_contract_citations_require_returned_evidence(self) -> None:
        scenario = {"prompt": "Use StageX."}
        recipe = {"stages": [{"ref": "StageX", "params": {}}]}
        trace = {
            "final_recipe": recipe,
            "tool_calls": [
                {
                    "verb": "cards",
                    "args": {"names": ["StageX"]},
                    "result": {"error": "lookup failed", "stage": "StageX"},
                },
                {
                    "verb": "validate",
                    "args": {"recipe": recipe},
                    "result": {
                        "runnable": True,
                        "status": "pass",
                        "semantic_review": {"status": "complete"},
                    },
                },
            ],
        }
        assert not _evidence_token_grounded("card:StageX", scenario, trace)
        assert not _evidence_token_grounded("contract:StageX", scenario, trace)

        trace["tool_calls"][0]["result"] = {
            "cards": [{"stage": "StageX", "card": {"stage_id": "StageX"}}]
        }
        trace["tool_calls"].insert(
            1,
            {
                "verb": "describe",
                "args": {"name": "StageX"},
                "result": {"stage": "StageX", "contract": {"reads": {}}},
            },
        )
        assert _evidence_token_grounded("card:StageX", scenario, trace)
        assert _evidence_token_grounded("contract:StageX", scenario, trace)

    def test_authoritative_judge_rejects_wrong_types_and_ranges(self) -> None:
        scenario = {"id": "semantic", "prompt": "Review this recipe."}
        trace = {"semantic_critique": {}, "final_recipe": {}, "tool_calls": []}
        invalid = judge(
            scenario,
            trace,
            model=lambda _prompt: (
                '{"items": [], "overall": 2.0, "grounded": "false", "notes": ""}'
            ),
        )
        assert invalid["authoritative"] is False
        assert invalid["grounded"] is False
        assert "validation_error" in invalid

        valid = judge(
            scenario,
            trace,
            model=lambda _prompt: (
                '{"items":[{"criterion":"intent","score":0.9,"met":true}],'
                '"overall":0.9,"grounded":true,"notes":"grounded"}'
            ),
        )
        assert valid["authoritative"] is True
        assert valid["overall"] == 0.9
        assert valid["grounded"] is True

    def test_expected_validation_rejects_failed_core_verdict(self, monkeypatch) -> None:
        scenario = {"expected_validation": {"status": "pass", "runnable": True}}
        trace = {"final_recipe": {"stages": [{"ref": "AnyStage", "params": {}}]}}

        from nemo_curator import audio_agent

        monkeypatch.setattr(
            audio_agent,
            "validate",
            lambda *_args, **_kwargs: {"status": "fail", "runnable": False},
        )
        ok, note, _ = _grade_validation(scenario, trace)
        assert not ok
        assert "runnable=False" in note
        assert "status=fail" in note

    def test_generic_judge_items_require_grounded_dimension_claims(self) -> None:
        scenario = {
            "prompt": "Consume the transformed output.",
            "semantic_expectations": {
                "critique": {
                    "min_items_by_section": {"transform_checks": 1},
                    "intent_status": "pass",
                    "mechanically_runnable": True,
                }
            }
        }
        recipe = {
            "stages": [
                {"ref": "TransformStage", "params": {}},
                {"ref": "ConsumerStage", "params": {}},
            ]
        }
        grounded = _critique_trace(scenario, recipe)
        items, is_grounded = _semantic_critique_items(scenario, grounded)
        assert is_grounded
        assert all(item["met"] for item in items)

        grounded["semantic_critique"]["transform_checks"][0]["evidence"] = ["trust me"]
        items, is_grounded = _semantic_critique_items(scenario, grounded)
        assert not is_grounded
        assert not items[0]["met"]


class TestSemanticScenarioMatrix:
    @staticmethod
    def _scenarios() -> list[dict]:
        return yaml.safe_load(_SEMANTIC_SCENARIOS.read_text(encoding="utf-8"))["scenarios"]

    def test_has_five_complete_pairs_covering_requested_boundaries(self) -> None:
        scenarios = self._scenarios()
        pairs: dict[str, set[str]] = {}
        for scenario in scenarios:
            pair = scenario["semantic_pair"]
            pairs.setdefault(pair["id"], set()).add(pair["side"])

        assert set(pairs) == {
            "parent_vs_child_field",
            "segment_vs_recording_quality",
            "aggregate_vs_row_metric",
            "used_vs_unused_transform",
            "explicit_wrong_key",
        }
        assert all(len(sides) == 2 for sides in pairs.values())
        assert set(_SEMANTIC_DEFAULT) == {scenario["id"] for scenario in scenarios}

    def test_live_capture_uses_real_repo_fixtures(self) -> None:
        for scenario in self._scenarios():
            fixture = (_ROOT / scenario["live_capture"]["fixture"]).resolve()
            assert fixture.is_file(), scenario["id"]
            request = _scenario_request(scenario)
            assert str(fixture) in request
            assert "manifest_path: REQUIRED" not in request
            assert "never fabricate a green verdict" in request

        scenario = self._scenarios()[0]
        recipe = _counterexample_recipe(
            self._scenarios()[1]["semantic_pair"]["mechanically_valid_counterexample"],
            scenario,
        )
        trace = _critique_trace(scenario, recipe)
        trace["final_recipe"]["stages"][0]["params"]["manifest_path"] = "REQUIRED"
        trace["tool_calls"][-1]["args"]["recipe"] = deepcopy(trace["final_recipe"])
        ok, problems, _ = _grade_semantic_expectations(scenario, trace)
        assert not ok
        assert any("concrete input fixture" in problem for problem in problems)

    def test_semantic_cli_refuses_a_missing_requested_scenario(
        self, monkeypatch
    ) -> None:
        from eval.audio import agent_runner

        monkeypatch.setattr(agent_runner, "_all_scenarios", lambda: {})
        assert agent_runner.main(["--semantic", "--mode", "template"]) == 2

    def test_each_pair_is_dual_with_the_same_mechanical_topology(self) -> None:
        scenarios = self._scenarios()
        by_pair: dict[str, list[dict]] = {}
        for scenario in scenarios:
            by_pair.setdefault(scenario["semantic_pair"]["id"], []).append(scenario)

        for pair in by_pair.values():
            left, right = pair
            left_refs = [
                stage["ref"]
                for stage in left["semantic_pair"]["mechanically_valid_counterexample"][
                    "stages"
                ]
            ]
            right_refs = [
                stage["ref"]
                for stage in right["semantic_pair"]["mechanically_valid_counterexample"][
                    "stages"
                ]
            ]
            assert left_refs == right_refs

            left_gold = _counterexample_recipe(
                right["semantic_pair"]["mechanically_valid_counterexample"],
                left,
            )
            right_gold = _counterexample_recipe(
                left["semantic_pair"]["mechanically_valid_counterexample"],
                right,
            )
            left_ok, left_problems, _ = _grade_semantic_expectations(
                left, _critique_trace(left, left_gold)
            )
            right_ok, right_problems, _ = _grade_semantic_expectations(
                right, _critique_trace(right, right_gold)
            )
            assert left_ok, f"{left['id']}: {left_problems}"
            assert right_ok, f"{right['id']}: {right_problems}"

            left_wrong, _, _ = _grade_semantic_expectations(
                left,
                _critique_trace(
                    left,
                    _counterexample_recipe(
                        left["semantic_pair"]["mechanically_valid_counterexample"],
                        left,
                    ),
                ),
            )
            right_wrong, _, _ = _grade_semantic_expectations(
                right,
                _critique_trace(
                    right,
                    _counterexample_recipe(
                        right["semantic_pair"]["mechanically_valid_counterexample"],
                        right,
                    ),
                ),
            )
            assert not left_wrong, left["id"]
            assert not right_wrong, right["id"]

    def test_every_declared_counterexample_is_mechanically_composable(self) -> None:
        from nemo_curator.audio_agent.recipe import Recipe, build_stages
        from nemo_curator.stages.audio._planning import validate_pipeline

        failures: list[str] = []
        for scenario in self._scenarios():
            counterexample = scenario["semantic_pair"]["mechanically_valid_counterexample"]
            recipe = Recipe.from_dict(_counterexample_recipe(counterexample))
            stages, build_issues = build_stages(recipe)
            if stages is None or build_issues:
                failures.append(f"{scenario['id']}: build issues={build_issues}")
                continue
            report = validate_pipeline(
                stages,
                initial_roles=set(counterexample.get("initial_roles") or []),
                initial_keys=set(counterexample.get("initial_keys") or []),
            )
            errors = [
                f"{issue.code}: {issue.message}"
                for issue in report.issues
                if issue.severity == "error"
            ]
            if errors:
                failures.append(f"{scenario['id']}: {errors}")

        assert not failures, "\n".join(failures)

    def test_full_validation_binds_each_counterexample_to_a_real_source(self) -> None:
        from nemo_curator import audio_agent

        source_codes = {
            "data_source_missing",
            "data_source_mismatch",
            "data_source_unsupported",
            "data_source_unreadable",
        }
        failures: list[str] = []
        for scenario in self._scenarios():
            counterexample = scenario["semantic_pair"]["mechanically_valid_counterexample"]
            verdict = audio_agent.validate(
                _counterexample_recipe(counterexample, scenario)
            )
            codes = {
                issue.get("code")
                for pool in ("issues", "card_violations", "gate_flags")
                for issue in verdict.get(pool, [])
            }
            bad = sorted(source_codes & codes)
            if bad:
                failures.append(f"{scenario['id']}: {bad}")
        assert not failures, "\n".join(failures)


class TestSemanticAggregateGate:
    @staticmethod
    def _passing_row(sid: str) -> dict:
        return {
            "id": sid,
            "status": "pass",
            "deterministic_status": "pass",
            "judge_authoritative": True,
            "semantic_judge_pass": True,
        }

    def test_authoritative_gate_requires_complete_passing_coverage(self) -> None:
        expected = ["a", "b"]
        complete = [self._passing_row("a"), self._passing_row("b")]
        assert _semantic_gate(
            complete,
            expected_ids=expected,
            authoritative_required=True,
        )["passed"] is True

        missing = _semantic_gate(
            complete[:1],
            expected_ids=expected,
            authoritative_required=True,
        )
        assert missing["passed"] is False
        assert missing["missing"] == ["b"]

        for field, value, report_key in (
            ("status", "blocked", "blocked"),
            ("deterministic_status", "fail", "deterministic_failures"),
            ("judge_authoritative", False, "non_authoritative"),
            ("semantic_judge_pass", False, "judge_failures"),
        ):
            rows = deepcopy(complete)
            rows[1][field] = value
            gate = _semantic_gate(
                rows,
                expected_ids=expected,
                authoritative_required=True,
            )
            assert gate["passed"] is False
            assert "b" in gate[report_key]

    def test_non_llm_mode_is_explicitly_diagnostic(self) -> None:
        gate = _semantic_gate(
            [],
            expected_ids=["missing"],
            authoritative_required=False,
        )
        assert gate["mode"] == "non_llm_diagnostic"
        assert gate["passed"] is None
