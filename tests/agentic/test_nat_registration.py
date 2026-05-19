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
"""Phase 3 smoke tests: confirm every ADV tool is registered with NAT and that
the JSON-in / JSON-out shim accepts the calling convention the React agent
will produce."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nat")


def test_every_tool_is_registered_with_nat() -> None:
    import nemo_curator.agentic.nat  # noqa: F401
    from nat.cli.type_registry import GlobalTypeRegistry

    from nemo_curator.agentic.tools import TOOLS

    fns = GlobalTypeRegistry.get().get_registered_functions()
    full_names = {f.full_type.split("/")[-1] for f in fns}
    for tool_name in TOOLS:
        assert f"curator_adv.{tool_name}" in full_names, (
            f"missing NAT registration for {tool_name}; "
            f"found: {sorted(n for n in full_names if n.startswith('curator_adv.'))}"
        )


def test_invoke_shim_accepts_json_string_and_dict() -> None:
    from nemo_curator.agentic.nat.register import _invoke
    from nemo_curator.agentic.tools import list_stages_tool

    out_dict = _invoke(list_stages_tool, {"limit": 1, "offset": 0})
    assert out_dict["limit"] == 1

    out_str = _invoke(list_stages_tool, json.dumps({"limit": 2, "offset": 0}))
    assert out_str["limit"] == 2


def test_invoke_shim_rejects_non_object_json() -> None:
    from nemo_curator.agentic.nat.register import _invoke
    from nemo_curator.agentic.tools import list_stages_tool

    with pytest.raises(ValueError, match="JSON object"):
        _invoke(list_stages_tool, "[1, 2, 3]")


def test_invoke_shim_rejects_unknown_kwarg() -> None:
    from nemo_curator.agentic.nat.register import _invoke
    from nemo_curator.agentic.tools import list_stages_tool

    with pytest.raises(ValueError, match="invalid arguments"):
        _invoke(list_stages_tool, {"not_a_real_arg": 1})


def test_capture_artifact_persists_compiled_yaml(tmp_path, monkeypatch) -> None:
    from nemo_curator.agentic.nat.register import _capture_artifact

    monkeypatch.setenv("ADV_AGENT_OUT_DIR", str(tmp_path))
    _capture_artifact("compile_ir", {"ir": {}}, "stages:\n  - foo")
    assert (tmp_path / "compiled.yaml").read_text() == "stages:\n  - foo"

    log = (tmp_path / "tool_calls.jsonl").read_text().strip().splitlines()
    assert len(log) == 1
    assert json.loads(log[0])["tool"] == "compile_ir"


def test_capture_artifact_persists_validation_results(tmp_path, monkeypatch) -> None:
    from nemo_curator.agentic.nat.register import _capture_artifact

    monkeypatch.setenv("ADV_AGENT_OUT_DIR", str(tmp_path))
    payload = {
        "ok": True,
        "findings": [{"severity": "info", "code": "ok"}],
        "ir": {"source": {"kind": "manifest", "uri": "/data"}, "sink": {}, "stages": []},
    }
    _capture_artifact("validate_ir", {}, payload)

    ir = json.loads((tmp_path / "ir.validated.json").read_text())
    assert ir["source"]["uri"] == "/data"
    findings = json.loads((tmp_path / "findings.json").read_text())
    assert findings[0]["code"] == "ok"


def test_capture_artifact_noop_when_env_unset(tmp_path, monkeypatch) -> None:
    from nemo_curator.agentic.nat.register import _capture_artifact

    monkeypatch.delenv("ADV_AGENT_OUT_DIR", raising=False)
    _capture_artifact("compile_ir", {}, "stages: []")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Coerce kwargs is the agent-facing JSON parser; it has to survive small LLM
# mistakes (trailing braces, code fences, ReAct prefixes) or the planner gets
# stuck in retry loops.
# ---------------------------------------------------------------------------


def test_coerce_kwargs_handles_trailing_extra_brace() -> None:
    from nemo_curator.agentic.nat.register import _coerce_kwargs

    out = _coerce_kwargs('{"intent": {"sample_rate": 48000}}}')
    assert out == {"intent": {"sample_rate": 48000}}


def test_coerce_kwargs_strips_action_input_prefix() -> None:
    from nemo_curator.agentic.nat.register import _coerce_kwargs

    out = _coerce_kwargs('Action Input: {"name": "VADSegmentationStage"}')
    assert out == {"name": "VADSegmentationStage"}


def test_coerce_kwargs_strips_code_fences() -> None:
    from nemo_curator.agentic.nat.register import _coerce_kwargs

    out = _coerce_kwargs('```json\n{"limit": 5}\n```')
    assert out == {"limit": 5}


def test_coerce_kwargs_ignores_trailing_observation_block() -> None:
    from nemo_curator.agentic.nat.register import _coerce_kwargs

    text = '{"intent": {"sample_rate": 48000}}\nObservation: {"supported": []}'
    out = _coerce_kwargs(text)
    assert out == {"intent": {"sample_rate": 48000}}


def test_coerce_kwargs_rejects_no_object() -> None:
    from nemo_curator.agentic.nat.register import _coerce_kwargs

    with pytest.raises(ValueError, match="contains no JSON object"):
        _coerce_kwargs("just some prose, no braces")
