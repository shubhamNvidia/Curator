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

"""Regressions for CLI input boundaries and shell-visible outcomes."""

from __future__ import annotations

import json

import pytest

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import cli


def test_goal_parser_accepts_free_text_and_mapping_json() -> None:
    assert cli._parse_goal("clean this corpus") == {"task": "clean this corpus"}
    assert cli._parse_goal('{"task": "clean", "language": "en"}') == {
        "task": "clean",
        "language": "en",
    }


@pytest.mark.parametrize("raw", ["[]", '["clean"]', '"clean"', "7", "null"])
def test_goal_parser_rejects_non_mapping_json(raw: str) -> None:
    with pytest.raises(ValueError, match="object mapping"):
        cli._parse_goal(raw)


def test_run_rejects_non_mapping_goal_before_calling_the_verb(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("stages: []\n", encoding="utf-8")
    called = False

    def unexpected_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("run must not receive an invalid goal")

    monkeypatch.setattr(aa, "run", unexpected_run)

    rc = cli.main(
        [
            "run",
            "--recipe",
            str(recipe),
            "--confirm",
            "--goal",
            "[1]",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert called is False
    assert "object mapping" in result["error"]


def test_calibration_loader_preserves_wrapper_metadata(tmp_path) -> None:
    wrapper = {
        "machine_fingerprint": "machine-a",
        "calibration": {
            "SomeStage": {
                "gpu_mem_gb": 3.5,
            }
        },
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(wrapper), encoding="utf-8")

    assert cli._calibration_arg(str(calibration)) == wrapper


@pytest.mark.parametrize("cmd", ["run", "continue", "report"])
def test_nested_unmet_acceptance_returns_nonzero(cmd: str) -> None:
    result = {
        "status": "completed",
        "acceptance": {
            "overall": "not_met",
        },
    }

    assert cli._result_exit_code(cmd, result) == 1
    assert cli._result_exit_code(
        cmd,
        {"status": "completed", "acceptance": {"overall": "met"}},
    ) == 0
    assert cli._result_exit_code(
        cmd,
        {"status": "completed", "acceptance": {}},
    ) == 0


def test_diagnose_cli_forwards_recipe_context_and_signals_action_required(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("stages: []\n", encoding="utf-8")
    expected = {"status": "action_required", "decision_required": True}
    called: dict[str, object] = {}

    def fake_diagnose(error: str, **kwargs: object) -> dict[str, object]:
        called["error"] = error
        called.update(kwargs)
        return expected

    monkeypatch.setattr(aa, "diagnose", fake_diagnose)
    rc = cli.main(
        [
            "diagnose",
            "--error",
            "CUDA error 222",
            "--recipe",
            str(recipe),
            "--operation",
            "smoke",
            "--attempted-actions",
            "upgrade_nvidia_driver",
        ]
    )

    assert json.loads(capsys.readouterr().out) == expected
    assert rc == 1
    assert called["error"] == "CUDA error 222"
    assert called["recipe"] == {"stages": []}
    assert called["operation"] == "smoke"
    assert called["attempted_actions"] == ["upgrade_nvidia_driver"]


def test_unknown_diagnosis_returns_nonzero() -> None:
    assert cli._result_exit_code("diagnose", {"status": "unknown"}) == 1
