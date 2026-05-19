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
"""Register the 12 ADV agentic tools with NVIDIA NeMo Agent Toolkit.

Each tool is exposed as a NAT *function* whose tool name starts with
``curator_adv.*`` so the agent runtime never collides with any other
plugin. Every tool accepts a single JSON-string argument (``input_json``)
that maps to the keyword arguments of the underlying core function. This
matches the React agent's typical calling convention and keeps the prompt
schemas easy to reason about.

The functions defined here are imported at package init time
(``nemo_curator.agentic.nat.__init__``) so the NAT global registry is
populated before any ``builder.get_tools(...)`` call.

Note: this module intentionally does NOT use ``from __future__ import
annotations``. NAT calls ``typing.get_type_hints`` on the inner closures
to derive each tool's input schema; deferred (string) annotations break
that resolution for ``Any`` / ``Union`` types.
"""

import datetime as _dt
import inspect
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from nemo_curator.agentic.tools import TOOLS

logger = logging.getLogger(__name__)

# Environment variable the CLI sets to redirect tool-call artifacts to disk.
# When unset (e.g. during unit tests), the capture is a no-op.
_OUT_DIR_ENV = "ADV_AGENT_OUT_DIR"


# ---------------------------------------------------------------------------
# Generic JSON-in / JSON-out tool wrapper.
# ---------------------------------------------------------------------------


def _coerce_kwargs(input_json: str | dict | None) -> dict[str, Any]:
    """Turn a raw agent argument into ``**kwargs`` for the underlying tool.

    The React agent in NAT historically passes tool input as a string. We
    accept either a JSON object (preferred), a JSON-encoded string, or an
    empty value (treated as no kwargs).

    To survive small LLM mistakes the parser is *forgiving* in two ways:

    - ``raw_decode`` is used to extract the **first** valid JSON object,
      so trailing characters (a stray ``}`` from brace miscount, an extra
      ``Observation:`` block, etc.) are silently ignored.
    - Common ReAct prefixes (``Action Input:``, code-fence markers) are
      stripped before parsing.
    """

    if input_json is None or input_json == "":
        return {}
    if isinstance(input_json, dict):
        return dict(input_json)
    if isinstance(input_json, str):
        text = _strip_react_prefixes(input_json).strip()
        if not text:
            return {}
        # Find the first ``{`` so we don't trip on leading prose.
        brace_at = text.find("{")
        if brace_at == -1:
            msg = (
                "tool input contains no JSON object; expected a "
                f"``{{...}}`` payload. Got: {input_json!r}"
            )
            raise ValueError(msg)
        candidate = text[brace_at:]
        try:
            decoded, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as exc:
            msg = (
                "tool input was not valid JSON; pass a JSON object containing "
                f"the tool's named arguments. Got: {input_json!r}"
            )
            raise ValueError(msg) from exc
        if not isinstance(decoded, dict):
            msg = (
                "tool input must decode to a JSON object (mapping arg name -> "
                f"value); got {type(decoded).__name__}"
            )
            raise ValueError(msg)
        leftover = candidate[end:].strip()
        if leftover and not leftover.startswith(("}", "```")):
            logger.debug("ignoring trailing tool-input text: %r", leftover[:80])
        return decoded
    msg = f"unsupported tool input type: {type(input_json).__name__}"
    raise TypeError(msg)


_REACT_PREFIXES = (
    "Action Input:",
    "Tool Input:",
    "Input:",
)


def _strip_react_prefixes(text: str) -> str:
    """Remove common ReAct scaffolding prefixes the model sometimes echoes."""

    stripped = text.lstrip()
    for prefix in _REACT_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    # Drop code-fence markers like ```json / ```
    if stripped.lstrip().startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        # Drop the closing fence if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped


def _invoke(tool: Callable[..., Any], input_json: str | dict | None) -> Any:
    kwargs = _coerce_kwargs(input_json)
    sig = inspect.signature(tool)

    # LangChain's tool-calling LLMs wrap the argument under the parameter
    # name ``input_json``. Unwrap one level whenever the underlying tool
    # does not expect that name itself.
    if (
        list(kwargs.keys()) == ["input_json"]
        and "input_json" not in sig.parameters
    ):
        kwargs = _coerce_kwargs(kwargs["input_json"])

    # Allow the agent to pass ``ir`` as a dict or as a JSON string.
    if "ir" in kwargs and isinstance(kwargs["ir"], str):
        try:
            kwargs["ir"] = json.loads(kwargs["ir"])
        except json.JSONDecodeError:
            # Treat as a path / inline JSON string handled by the core helper.
            pass

    try:
        bound = sig.bind_partial(**kwargs)
        bound.apply_defaults()
    except TypeError as exc:
        msg = f"invalid arguments for tool {tool.__name__}: {exc}"
        raise ValueError(msg) from exc

    return tool(**bound.arguments)


# ---------------------------------------------------------------------------
# Per-tool config classes (one per tool, so NAT can validate names).
# ---------------------------------------------------------------------------


_DESCRIPTIONS: dict[str, str] = {
    "profile_source": (
        "Layer-1 dataset profiler. Returns a JSON DatasetCard summarizing "
        "sample rates, channel counts, duration distribution, and decode "
        "failure rate. Input JSON: {\"source_uri\": \"...\", \"kind\": "
        "\"manifest|directory\", \"sample_limit\": 64}. Call this exactly "
        "once per prompt before planning."
    ),
    "capability_search": (
        "Find every stage that satisfies a CapabilityTag. Input JSON: "
        "{\"capability\": \"asr\", \"commercial_only\": false, "
        "\"include_also_handles\": true}. Returns a list of stage summaries."
    ),
    "stage_inspect": (
        "Fetch one stage card (parameters, defaults, ranges, resource hints, "
        "license). Input JSON: {\"name\": \"VADSegmentationStage\"}. Always "
        "call this before choosing a stage's parameter values."
    ),
    "required_capabilities": (
        "Translate an IntentCategories JSON into the list of CapabilityTags "
        "the pipeline must satisfy. Input JSON: {\"intent\": {...}}."
    ),
    "gap_report": (
        "Cross-reference an IntentCategories JSON against the catalog. "
        "Input JSON: {\"intent\": {...}}. Returns {\"supported\": [...], "
        "\"gaps\": [...], \"has_gaps\": bool}. If has_gaps==true the agent "
        "MUST refuse the prompt and report the missing capability tags."
    ),
    "validate_ir": (
        "Run the deterministic 8-check static validator over a PipelineIR. "
        "Input JSON: {\"ir\": {...}, \"mutate\": true}. Returns "
        "{\"ok\": bool, \"findings\": [...], \"ir\": {...}, "
        "\"auto_inserted\": [...]}."
    ),
    "compile_ir": (
        "Compile a validated PipelineIR into the canonical stages: YAML the "
        "runtime consumes. Input JSON: {\"ir\": {...}}. Returns YAML text."
    ),
    "run_ir": (
        "Execute (or dry-run) a PipelineIR end-to-end. Input JSON: "
        "{\"ir\": {...}, \"target_dir\": \"/tmp/out\", \"dry_run\": false, "
        "\"enable_cache\": true}. Returns a RunCard JSON."
    ),
    "dry_run_ir": (
        "Convenience wrapper that always forces dry-run. Input JSON: "
        "{\"ir\": {...}}."
    ),
    "deterministic_critic": (
        "LLM-free critic over a completed RunCard. Input JSON: "
        "{\"run_card\": {...}, \"intent\": {...}, \"input_card\": {...}, "
        "\"output_card\": {...}}. Returns a CriticReport JSON."
    ),
    "cache_gc": (
        "LRU-evict the per-stage cache. Input JSON: "
        "{\"target_dir\": \"/tmp/out\", \"max_bytes\": 53687091200}."
    ),
    "list_stages": (
        "Paginated stage catalog overview. Input JSON: "
        "{\"category\": \"filter\", \"commercial_only\": false, "
        "\"limit\": 64, \"offset\": 0}."
    ),
}


def _make_config(tool_name: str) -> type[FunctionBaseConfig]:
    """Synthesize a ``FunctionBaseConfig`` subclass for one tool."""

    nat_name = f"curator_adv.{tool_name}"
    description = _DESCRIPTIONS[tool_name]

    cls = type(
        f"CuratorAdv{tool_name.title().replace('_', '')}Config",
        (FunctionBaseConfig,),
        {
            "__module__": __name__,
            "__doc__": description,
            "__annotations__": {"description": str},
            "description": Field(default=description, description="Tool description."),
        },
        name=nat_name,
    )
    return cls


def _build_register_fn(
    tool_name: str,
    cfg_cls: type[FunctionBaseConfig],
) -> Callable[..., Any]:
    """Wrap one core tool as a NAT ``@register_function``."""

    tool = TOOLS[tool_name]

    @register_function(config_type=cfg_cls)
    async def _register(config, builder: Builder):  # noqa: ARG001
        # ``input_json`` is typed ``object`` so the React agent can pass either
        # a JSON string (text-format ReAct) or a Python dict (tool-calling
        # LLMs emit dicts directly). ``object`` plays nicely with NAT's
        # pydantic-derived input schema AND with isinstance-based validators
        # — using ``Any`` here emits noisy "Any cannot be used with isinstance"
        # warnings during tool invocation.
        async def _arun(input_json: object = "") -> str:
            try:
                result = _invoke(tool, input_json)
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool %s failed: %s", tool_name, exc)
                err_payload = {"error": str(exc), "tool": tool_name}
                _capture_artifact(tool_name, input_json, err_payload)
                return json.dumps(err_payload)
            _capture_artifact(tool_name, input_json, result)
            return _to_json(result)

        yield FunctionInfo.from_fn(_arun, description=config.description)

    return _register


def _to_json(value: Any) -> str:
    """Best-effort JSON serialization for whatever the tool returned."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=_default_encoder, ensure_ascii=False)
    except TypeError:
        return json.dumps(repr(value))


def _capture_artifact(tool_name: str, input_value: object, result: Any) -> None:
    """Persist interesting tool outputs to ``$ADV_AGENT_OUT_DIR`` (if set).

    Three slots are recognized because the rest of the agentic layer expects
    these specific filenames:

    - ``compile_ir``  → ``compiled.yaml``
    - ``validate_ir`` → ``ir.validated.json`` + ``findings.json``
    - ``run_ir``      → ``run_artifacts/`` (copied from the runtime target_dir)

    Every tool call (including failures) appends one JSONL line to
    ``tool_calls.jsonl`` so the run is fully reconstructible after the fact.
    Capture is best-effort: any disk error is logged at WARNING and the
    underlying agent run keeps going.
    """

    out = os.environ.get(_OUT_DIR_ENV)
    if not out:
        return
    try:
        out_dir = Path(out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _append_call_log(out_dir, tool_name, input_value, result)

        if tool_name == "compile_ir" and isinstance(result, str) and result.strip():
            (out_dir / "compiled.yaml").write_text(result, encoding="utf-8")
        elif tool_name == "validate_ir" and isinstance(result, dict):
            if "ir" in result:
                (out_dir / "ir.validated.json").write_text(
                    json.dumps(result["ir"], indent=2, default=_default_encoder),
                    encoding="utf-8",
                )
            if "findings" in result:
                (out_dir / "findings.json").write_text(
                    json.dumps(result["findings"], indent=2, default=_default_encoder),
                    encoding="utf-8",
                )
        elif tool_name == "run_ir" and isinstance(result, dict):
            src = result.get("target_dir")
            if src and Path(src).is_dir():
                adv_src = Path(src) / ".adv"
                adv_dst = out_dir / "run_artifacts"
                if adv_src.is_dir():
                    shutil.copytree(adv_src, adv_dst, dirs_exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("artifact capture for %s failed: %s", tool_name, exc)


def _append_call_log(out_dir: Path, tool_name: str, input_value: object, result: Any) -> None:
    record = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
        "tool": tool_name,
        "input": _safe_record(input_value),
        "result": _safe_record(result),
    }
    with (out_dir / "tool_calls.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_default_encoder, ensure_ascii=False) + "\n")


def _safe_record(value: Any) -> Any:
    """Trim very large strings for the audit log; keep dicts/lists intact."""
    if isinstance(value, str):
        return value if len(value) <= 32_000 else value[:32_000] + "...[truncated]"
    return value


def _default_encoder(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


# ---------------------------------------------------------------------------
# Eager registration (runs once on import).
# ---------------------------------------------------------------------------


_REGISTRATIONS: dict[str, Callable[..., Any]] = {}

for _name in TOOLS:
    _cfg = _make_config(_name)
    _REGISTRATIONS[_name] = _build_register_fn(_name, _cfg)


__all__ = ["_DESCRIPTIONS", "_REGISTRATIONS"]
