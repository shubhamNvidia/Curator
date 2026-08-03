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

"""Capture LLM-plane agent traces for scenarios (AGENT_TEST_PLAN.md 6.1).

Drives a real agent over each scenario prompt with the Cursor SDK, giving it the
audio-agent verbs (as an inline stdio MCP server) and the audio-curation workflow
(SKILL.md, injected as a preamble since SDK setting-sources default to inline).
Records the tool-call sequence + final recipe + explanation into a trace JSON that
`trace_check.py` / `judge.py` grade.

    export CURSOR_API_KEY=cursor_...
    python -m eval.audio.agent_runner --batch                 # curated cross-level set
    python -m eval.audio.agent_runner --semantic              # paired intent regressions
    python -m eval.audio.agent_runner --id L02_readspeech_quality --mode sdk
    python -m eval.audio.agent_runner --id L04_good_audio --mode template   # manual fallback

Capture is decoupled from grading, so a template (hand-filled) trace grades the
same way as an SDK-captured one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_TRACES_DIR = os.path.join(_HERE, "traces")
_SCENARIOS_DIR = os.path.join(_HERE, "scenarios")
_SKILL = os.path.join(_ROOT, ".claude", "skills", "audio-curation", "SKILL.md")

# Verbs the deterministic core exposes; used to normalize MCP tool names -> verb.
_VERBS = (
    "discover", "describe", "catalog_tree", "cards", "context", "resolve",
    "validate", "smoke", "run", "report", "verify", "runs", "plan_continuation", "calibrate",
    "reuse_scan", "reindex", "diagnose", "doctor",
)

# Default curated batch: one anchor per level (L01..L14) + two high-signal extras.
_BATCH_DEFAULT = [
    "L01_duration", "L02_readspeech_quality", "L03_per_segment_quality", "L04_good_audio",
    "L05_wer_no_reference", "L06_transcribe_no_audio", "L07_quality_metric_choice",
    "L08_add_transcripts", "L09_file_export", "L10_custom_score_key", "L11_task_type_break",
    "L12_emotion_labeling", "L13_unknown_stage", "L14_use_new_stage",
    "L02_transcribe_export", "L11_tensor_into_sink",
]

# Paired intent regressions whose counterexamples are mechanically valid.  Keep
# these separate from the cross-level batch so callers can run the higher-cost
# semantic suite explicitly (``--semantic``) without changing the historical
# batch size.
_SEMANTIC_DEFAULT = [
    "S15_parent_count_filter",
    "S15_child_track_filter",
    "S15_segment_quality",
    "S15_recording_quality",
    "S15_row_quality_gate",
    "S15_aggregate_quality_gate",
    "S15_resample_copy",
    "S15_resample_for_asr",
    "S15_filter_sigmos_key",
    "S15_filter_utmos_key",
]


def _all_scenarios() -> dict:
    import yaml

    out = {}
    for f in sorted(glob.glob(os.path.join(_SCENARIOS_DIR, "*.yaml"))):
        for s in (yaml.safe_load(open(f, encoding="utf-8")) or {}).get("scenarios", []):
            out[s["id"]] = s
    return out


def find_scenario(sid: str) -> dict:
    scs = _all_scenarios()
    if sid not in scs:
        msg = f"scenario id {sid!r} not found under {_SCENARIOS_DIR}"
        raise KeyError(msg)
    return scs[sid]


def _scenario_fixture(scenario: dict) -> str | None:
    """Resolve an eval-owned fixture and refuse paths outside the repository."""
    live = scenario.get("live_capture") or {}
    rel = live.get("fixture") if isinstance(live, dict) else None
    if not isinstance(rel, str) or not rel.strip():
        return None
    path = os.path.realpath(os.path.join(_ROOT, rel))
    root = os.path.realpath(_ROOT)
    if os.path.commonpath([root, path]) != root:
        msg = f"scenario fixture escapes the repository: {rel!r}"
        raise ValueError(msg)
    if not os.path.isfile(path):
        msg = f"scenario fixture does not exist: {path}"
        raise FileNotFoundError(msg)
    return path


def _scenario_request(scenario: dict) -> str:
    """Add concrete eval facts without turning them into user preferences."""
    request = str(scenario.get("prompt") or "")
    fixture = _scenario_fixture(scenario)
    if fixture is None:
        return request
    live = scenario.get("live_capture") or {}
    environment = str(live.get("environment") or "real_host")
    return (
        f"{request}\n\n"
        "EVALUATION CONTEXT (facts, not user preferences):\n"
        f"- The exact input manifest is {fixture}. Configure the recipe's source "
        "stage with this path and, if passing a data assertion, use the same path.\n"
        "- Run doctor before validation. Use the real validation/environment "
        "result; never fabricate a green verdict, change execution target, or "
        "silently switch device/model to satisfy this eval.\n"
        f"- Environment strategy: {environment}. If the selected recipe is blocked "
        "on this host, report that honestly; the authoritative regression gate "
        "will record an environment-blocked capture rather than a semantic pass."
    )


def _norm_verb(name: str) -> str:
    """Map an MCP tool name (possibly prefixed) to a bare verb."""
    n = str(name or "").strip()
    if not n:
        return "tool"
    for v in _VERBS:
        if n == v or n.endswith(v) or n.endswith("." + v) or n.endswith("_" + v):
            return v
    return n.split(".")[-1]


def _mcp_result(res: object) -> object:
    """Unwrap an MCP tool result ({value:{content:[{text:{text: <json>}}]}}) to the
    parsed inner object, truncating large payloads so traces stay lean."""
    if res is None:
        return None
    try:
        val = res.get("value", res) if isinstance(res, dict) else res
        content = val.get("content") if isinstance(val, dict) else None
        if isinstance(content, list) and content:
            txt = content[0].get("text")
            if isinstance(txt, dict):
                txt = txt.get("text")
            if isinstance(txt, str):
                try:
                    return json.loads(txt)
                except Exception:  # noqa: BLE001
                    return txt[:500]
        return val if not isinstance(val, (dict, list)) else "<result>"
    except Exception:  # noqa: BLE001
        return None


def _extract_mcp(msg: object) -> tuple[str, dict, object]:
    """Return (verb, args, result) from an SDKToolUseMessage, handling the generic
    'mcp' wrapper where the real tool name lives in args['toolName']."""
    name = getattr(msg, "name", "") or ""
    raw = getattr(msg, "args", None) or {}
    if name == "mcp" or (isinstance(raw, dict) and "toolName" in raw):
        verb = _norm_verb(raw.get("toolName", name))
        args = raw.get("args", {})
    else:
        verb = _norm_verb(name)
        args = raw
    return verb, args, _mcp_result(getattr(msg, "result", None))


def _parse_recipe(text: str) -> dict | None:
    """Extract the final recipe from a fenced ```recipe / ```yaml block."""
    import yaml

    for pat in (r"```recipe\s*(.*?)```", r"```yaml\s*(.*?)```"):
        m = re.search(pat, text or "", re.DOTALL)
        if m:
            try:
                doc = yaml.safe_load(m.group(1))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(doc, dict) and doc.get("stages"):
                return doc
    return None


def _parse_semantic_critique(text: str) -> dict | None:
    """Extract the host critic's structured semantic-critique artifact.

    ``validate.semantic_review`` is the deterministic evidence packet; this is
    the distinct host-LLM judgement required by the audio-curation skill.
    """
    import yaml

    for pat in (
        r"```semantic-critique\s*(.*?)```",
        r"```semantic_critique\s*(.*?)```",
    ):
        m = re.search(pat, text or "", re.DOTALL)
        if not m:
            continue
        try:
            doc = yaml.safe_load(m.group(1))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(doc, dict) and isinstance(doc.get("semantic_critique"), dict):
            doc = doc["semantic_critique"]
        if isinstance(doc, dict):
            return doc
    return None


def _preamble() -> str:
    try:
        skill = open(_SKILL, encoding="utf-8").read()
    except Exception:  # noqa: BLE001
        skill = "(SKILL.md not found; follow: discover/context/cards -> resolve -> validate; never invent params.)"
    return (
        "You are the host planner for NeMo Curator audio curation. Follow this workflow exactly.\n"
        "Use ONLY the nemo_curator.audio_agent MCP tools (discover/context/cards/resolve/validate/...).\n"
        "Never invent stage or parameter names. Resolve outcome->threshold with `resolve` (never hand-pick).\n\n"
        "OUTPUT CONTRACT (follow strictly):\n"
        "1. Discover stages, read the relevant cards, `resolve` any thresholds, and `validate` your recipe.\n"
        "2. If (and only if) the request is ambiguous on a material, preference-dependent point, ask ONE\n"
        "   short user-facing question and STOP - do NOT emit a recipe.\n"
        "3. If the request is unsupported or impossible, say so briefly and STOP - do NOT emit a recipe.\n"
        "4. OTHERWISE, after mechanical validation, inspect the Verdict's deterministic semantic_review\n"
        "   evidence packet and critique the DRAFT against the user's intent. Emit the host judgement as:\n"
        "```semantic-critique\n"
        "mechanically_runnable: true\n"
        "recipe_config_hash: <copy validate.semantic_review.recipe.config_hash exactly>\n"
        "intent_status: pass\n"
        "stage_reviews:\n"
        "  - {stage: StageName, finding: \"why this stage serves the goal\", evidence: [\"goal:user-request\", \"card:StageName\"]}\n"
        "field_reviews:\n"
        "  - {field: output_key, producer: ProducerStage, finding: \"meaning and granularity here\", evidence: [\"contract:ProducerStage\", \"recipe:ConsumerStage\"]}\n"
        "behavior_checks: []\n"
        "transform_checks: []\n"
        "model_checks: []\n"
        "assumptions_or_questions: []\n"
        "```\n"
        "   Use every section required by SKILL.md. Each non-empty review item must have a substantive\n"
        "   finding and concise source:locator evidence tokens (card:, contract:, goal:, recipe:, validate:,\n"
        "   data:, or smoke:). Use intent_status revise/ask instead of claiming validate proved intent.\n"
        "5. Then end your final message with the runnable pipeline as a fenced block EXACTLY\n"
        "   in this form (emit it even for a single-stage pipeline; never describe the pipeline only in prose):\n"
        "```recipe\n"
        "stages:\n"
        "  - {ref: ManifestReader, params: {manifest_path: <exact path from EVALUATION CONTEXT>}}\n"
        "  - {ref: GetAudioDurationStage, params: {}}\n"
        "```\n"
        "followed by one short paragraph of rationale (which metric/module and why, and any assumption).\n\n"
        "==== audio-curation SKILL ====\n" + skill + "\n==== end SKILL ====\n"
    )


def template_trace(scenario: dict) -> dict:
    """A trace skeleton to fill in by hand after running the agent in-IDE."""
    fixture = _scenario_fixture(scenario)
    return {
        "scenario_id": scenario.get("id"),
        "prompt": scenario.get("prompt"),
        "captured_by": "manual",
        "tool_calls": [
            {"verb": "doctor", "args": {}, "result": {"status": "<fill from real host>"}},
            {"verb": "context", "args": {"goal": {}, "data": fixture or "<path>"}},
            {"verb": "cards", "args": {"category": "<category>"}},
            {"clarification": "<the user-facing question you asked, or delete this line>"},
            {"verb": "resolve", "args": {"stage": "<Stage>", "label": "<outcome>"}},
            {"verb": "validate", "args": {"recipe": "<recipe>"}, "result": {"runnable": None, "status": None}},
        ],
        "final_recipe": {"stages": [{"ref": "<Stage>", "params": {}}]},
        "semantic_critique": {
            "mechanically_runnable": True,
            "recipe_config_hash": "<copy validate.semantic_review.recipe.config_hash>",
            "intent_status": "pass",
            "stage_reviews": [
                {
                    "stage": "<Stage>",
                    "finding": "<why this stage serves the user goal>",
                    "evidence": ["goal:user-request", "card:<Stage>"],
                }
            ],
            "field_reviews": [],
            "behavior_checks": [],
            "transform_checks": [],
            "model_checks": [],
            "assumptions_or_questions": [],
        },
        "refused": False,
        "continuation": None,
        "explanation": "<paste the agent's user-facing rationale here>",
    }


def run_via_sdk(scenario: dict, *, model: str = "auto", api_key: str | None = None) -> dict:
    """Drive a real agent over the scenario via the Cursor SDK; return a trace dict.

    On a startup failure (auth/config/network) returns a trace flagged
    ``blocked=True`` so a batch can continue and report the blocker.
    """
    from cursor_sdk import (
        Agent,
        AgentOptions,
        CursorAgentError,
        LocalAgentOptions,
        SDKAssistantMessage,
        SDKToolUseMessage,
        StdioMcpServerConfig,
    )

    key = api_key or os.environ.get("CURSOR_API_KEY")
    mcp = {
        "audio-agent": StdioMcpServerConfig(
            command=os.path.join(_ROOT, ".venv", "bin", "python"),
            args=["-m", "nemo_curator.audio_agent.mcp_server"],
            cwd=_ROOT,
            env=dict(os.environ),
        )
    }
    opts = AgentOptions(
        model=model, api_key=key,
        local=LocalAgentOptions(cwd=_ROOT),
        mcp_servers=mcp,
    )
    prompt = _preamble() + "\n\nUSER REQUEST: " + _scenario_request(scenario)

    calls: list[dict] = []
    texts: list[str] = []
    status = None
    final_text = ""
    try:
        with Agent.create(opts) as agent:
            run = agent.send(prompt)
            for msg in run.messages():
                if isinstance(msg, SDKToolUseMessage):
                    if getattr(msg, "result", None) is None:
                        continue  # 'started' event; keep only the completed one (dedupe)
                    verb, targs, tres = _extract_mcp(msg)
                    calls.append({"verb": verb, "args": targs, "result": tres})
                elif isinstance(msg, SDKAssistantMessage):
                    txt = getattr(getattr(msg, "message", None), "text", None)
                    if txt:
                        texts.append(txt)
            result = run.wait()
            status = getattr(result, "status", None)
            try:
                final_text = run.text() or ""
            except Exception:  # noqa: BLE001
                final_text = ""
    except CursorAgentError as e:
        return {
            "scenario_id": scenario.get("id"), "prompt": scenario.get("prompt"),
            "captured_by": f"cursor_sdk:{model}", "blocked": True,
            "error": f"startup failed (auth/config/network): {e}",
            "tool_calls": [], "final_recipe": None, "refused": True, "explanation": "",
        }

    text = final_text or "\n".join(texts)
    recipe = _parse_recipe(text)
    semantic_critique = _parse_semantic_critique(text)
    asked = recipe is None and "?" in (text or "")
    trace = {
        "scenario_id": scenario.get("id"), "prompt": scenario.get("prompt"),
        "captured_by": f"cursor_sdk:{model}", "run_status": status,
        "tool_calls": calls, "final_recipe": recipe,
        "semantic_critique": semantic_critique,
        "refused": recipe is None, "explanation": text,
    }
    if asked:
        trace["tool_calls"].append({"clarification": text.strip().splitlines()[-1] if text.strip() else "asked"})
    return trace


def _write(trace: dict, out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description="Capture agent traces for scenarios")
    ap.add_argument("--id", help="single scenario id")
    ap.add_argument("--batch", action="store_true", help="run the curated cross-level set")
    ap.add_argument("--semantic", action="store_true", help="run the paired semantic-intent regression set")
    ap.add_argument("--all", action="store_true", help="with --batch, run every scenario")
    ap.add_argument("--ids", help="comma-separated scenario ids (overrides the default batch set)")
    ap.add_argument("--mode", choices=["sdk", "template"], default="sdk")
    ap.add_argument("--model", default="auto")
    ap.add_argument(
        "--trace-dir",
        default=_TRACES_DIR,
        help="directory for batch traces (use a fresh directory for regression gates)",
    )
    ap.add_argument("--out", default=None, help="output path for --id (default traces/<id>.json)")
    args = ap.parse_args(argv)

    scenarios = _all_scenarios()
    if args.semantic and args.all:
        ap.error("--all applies to --batch; use --semantic by itself for the paired suite")

    if args.batch or args.semantic or args.ids:
        if args.ids:
            ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        elif args.semantic:
            ids = list(_SEMANTIC_DEFAULT)
        elif args.all:
            ids = list(scenarios)
        else:
            ids = [i for i in _BATCH_DEFAULT if i in scenarios]
        blocked = 0
        missing = [sid for sid in ids if sid not in scenarios]
        if missing:
            print(f"[error] unknown scenario ids: {missing}")
            return 2
        for sid in ids:
            sc = scenarios[sid]
            trace = template_trace(sc) if args.mode == "template" else run_via_sdk(sc, model=args.model)
            out = os.path.join(args.trace_dir, f"{sid}.json")
            _write(trace, out)
            tag = "BLOCKED" if trace.get("blocked") else ("refused" if trace.get("refused") else "captured")
            if trace.get("blocked"):
                blocked += 1
            print(f"[{tag}] {sid} -> {out}" + (f"  ({trace.get('error')})" if trace.get("blocked") else ""))
        print(json.dumps({"batch": len(ids), "blocked": blocked, "mode": args.mode}))
        return 1 if blocked and blocked == len(ids) else 0

    if not args.id:
        ap.error("provide --id, --batch, --semantic, or --ids")
    scenario = find_scenario(args.id)
    trace = template_trace(scenario) if args.mode == "template" else run_via_sdk(scenario, model=args.model)
    out = args.out or os.path.join(args.trace_dir, f"{args.id}.json")
    _write(trace, out)
    print(f"[agent_runner] wrote {args.mode} trace -> {out}")
    if trace.get("blocked"):
        print(f"[blocked] {trace.get('error')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
