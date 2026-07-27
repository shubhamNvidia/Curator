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

"""CLI adapter over the audio-agent verbs — the shell/notebook surface.

Every subcommand maps 1:1 to a verb and prints a JSON result, so a host agent
(or a human) can drive the ``discover -> route -> validate -> smoke -> confirm ->
run -> report`` loop from a terminal. The same core also backs the MCP adapter.

    python -m nemo_curator.audio_agent discover
    python -m nemo_curator.audio_agent catalog-tree
    python -m nemo_curator.audio_agent cards --category quality
    python -m nemo_curator.audio_agent validate --recipe recipe.yaml --data data.jsonl
    python -m nemo_curator.audio_agent smoke   --recipe recipe.yaml --sample 10
    python -m nemo_curator.audio_agent run     --recipe recipe.yaml --confirm <hash>
    python -m nemo_curator.audio_agent report  --output out/ --data data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _load_recipe(path: str) -> dict[str, Any]:
    import yaml

    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    return yaml.safe_load(text)


def _load_doc(path: str | None) -> Any:  # noqa: ANN401
    """Load a YAML/JSON doc from a file path (or ``-`` for stdin); ``None`` -> None."""
    if not path:
        return None
    import yaml

    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    return yaml.safe_load(text)


def _criteria_list(doc: Any) -> list[dict[str, Any]] | None:  # noqa: ANN401
    """Accept a bare list, or a mapping carrying ``acceptance_criteria`` / ``criteria``.

    Fail LOUD (ValueError) when a non-empty doc doesn't yield a criteria list -- a common
    mistake is a top-level mapping like ``{output_completeness: ..., yield: ...}`` -- so a
    malformed criteria file is reported instead of being silently ignored (which used to
    make ``validate``/``verify`` skip the contract without any warning).
    """
    if doc is None:
        return None
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        crit = doc.get("acceptance_criteria")
        if crit is None:
            crit = doc.get("criteria")
        if crit is None:
            if not doc:  # genuinely empty mapping -> no criteria
                return None
            msg = (
                "acceptance criteria not recognized: expected a YAML/JSON LIST of criteria, "
                "or a mapping with an 'acceptance_criteria' (or 'criteria') key holding that "
                f"list; got a mapping with top-level keys {sorted(doc)!r}. Each criterion is "
                "{id, type, check:{field,op,value}, severity}; e.g.\n"
                "  acceptance_criteria:\n"
                "    - {id: dur, type: output_completeness, compiles_to: duration, severity: must}\n"
                "    - {id: keep, type: yield, kind: absolute, check: {op: '==', value: 4}, severity: must}"
            )
            raise ValueError(msg)
        if not isinstance(crit, list):
            msg = f"'acceptance_criteria' must be a list of criterion mappings, got {type(crit).__name__}"
            raise ValueError(msg)
        return crit
    msg = f"acceptance criteria must be a list or a mapping with 'acceptance_criteria', got {type(doc).__name__}"
    raise ValueError(msg)


def _calibration_arg(path: str | None) -> dict[str, Any] | None:
    """Accept a bare ``{stage: {...}}`` calibration or the ``{calibration: {...}}`` wrapper."""
    doc = _load_doc(path)
    if isinstance(doc, dict) and "calibration" in doc:
        return doc["calibration"]
    return doc


def _emit(obj: Any) -> None:  # noqa: ANN401
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _parse_goal(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"task": raw}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nemo_curator.audio_agent", description="Audio Agent (P1) tool surface")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("discover", help="list agent-ready audio stages with category + one-liner")
    sub.add_parser("catalog-tree", help="L0 category tree for coarse-to-fine routing")

    d = sub.add_parser("describe", help="static contract (+ card) for one stage")
    d.add_argument("name")

    c = sub.add_parser("cards", help="L1 (one-liners for a category) or L2 (full cards for names)")
    c.add_argument("--category")
    c.add_argument("--names", nargs="*")

    ctx = sub.add_parser("context", help="assemble a PlanningContext for the router/planner")
    ctx.add_argument("--goal", help="JSON goal spec, or a free-text task string")
    ctx.add_argument("--data")
    ctx.add_argument("--stages", nargs="*")
    ctx.add_argument("--roles", nargs="*")

    v = sub.add_parser("validate", help="validate a recipe (roles/keys/cards/gates)")
    v.add_argument("--recipe", required=True, help="path to a recipe YAML/JSON (or - for stdin)")
    v.add_argument("--data")
    v.add_argument("--expected-outputs", nargs="*", help="semantic output roles the user asked for (output-completeness)")
    v.add_argument("--acceptance-criteria", help="path to acceptance criteria YAML/JSON (list or {acceptance_criteria: [...]})")
    v.add_argument("--request-type", help="goal/request kind (e.g. filter, transcribe) for request-type sanity")

    s = sub.add_parser("smoke", help="bounded run for evidence")
    s.add_argument("--recipe", required=True)
    s.add_argument("--sample", type=int, default=10)
    s.add_argument("--data")
    s.add_argument("--output-dir")
    s.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")
    s.add_argument("--calibration", help="path to a calibration JSON from a prior smoke (1C.2)")

    r = sub.add_parser("run", help="confirm-gated full run (0 silent runs)")
    r.add_argument("--recipe", required=True)
    r.add_argument("--confirm", nargs="?", const=True, default=False,
                   help="pass the recipe config_hash (integrity) or bare --confirm")
    r.add_argument("--data")
    r.add_argument("--output-dir")
    r.add_argument("--checkpoint-path")
    r.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")
    r.add_argument("--smoke-token", help="smoke-evidence token from a prior smoke (required if AUDIO_AGENT_REQUIRE_SMOKE is set)")
    r.add_argument("--calibration", help="path to a calibration JSON from a prior smoke (1C.2)")

    rp = sub.add_parser("report", help="post-hoc report from an output manifest/dir")
    rp.add_argument("--output", required=True)
    rp.add_argument("--recipe")
    rp.add_argument("--data")

    vf = sub.add_parser("verify", help="verify acceptance criteria against evidence -> AcceptanceReport")
    vf.add_argument("--criteria", required=True, help="acceptance criteria YAML/JSON (list or {acceptance_criteria: [...]}; - for stdin)")
    vf.add_argument("--evidence", help="evidence YAML/JSON (produced_roles/metrics/retained/...); - for stdin")
    vf.add_argument("--frozen-criteria", help="the confirmed contract, for the honesty guard (YAML/JSON)")
    vf.add_argument("--recipe", dest="verify_recipe", help="recipe carrying acceptance_criteria (alt source for the honesty guard)")

    rs = sub.add_parser("resolve", help="resolve an outcome (label/use_case/explicit) to concrete stage config (1A.2)")
    rs.add_argument("--stage", required=True)
    rs.add_argument("--label", help="outcome label, e.g. studio / transcription_grade")
    rs.add_argument("--use-case", help="named card preset, e.g. tts_reference")
    rs.add_argument("--explicit", help="JSON object of {param: value}")
    rs.add_argument("--data-driven", action="store_true", help="enable Path B (deferred; affects relative-objective asks)")

    ru = sub.add_parser("runs", help="list local run records (provenance) or show one by id")
    ru.add_argument("--run-id", help="show a single run record")

    cont = sub.add_parser("continue", help="plan a follow-up run incrementally against a prior run (reuse where safe)")
    cont.add_argument("--recipe", required=True, help="the follow-up recipe (or - for stdin)")
    cont.add_argument("--parent-run-id", required=True, help="the prior run to continue from")
    cont.add_argument("--data", help="the source dataset (for the same-data fingerprint guard)")

    cal = sub.add_parser("calibrate", help="extract measured per-stage resources from a smoke report (1C.2)")
    cal.add_argument("--smoke", required=True, help="path to a smoke-result JSON (or - for stdin)")
    return p


def main(argv: list[str] | None = None) -> int:
    from nemo_curator import audio_agent as aa

    args = build_parser().parse_args(argv)
    cmd = args.cmd

    try:
        if cmd == "discover":
            _emit(aa.discover())
        elif cmd == "catalog-tree":
            _emit(aa.catalog_tree())
        elif cmd == "describe":
            _emit(aa.describe(args.name))
        elif cmd == "cards":
            _emit(aa.cards(category=args.category, names=args.names))
        elif cmd == "context":
            _emit(aa.context(_parse_goal(args.goal), data=args.data, stages=args.stages, roles=args.roles))
        elif cmd == "validate":
            _emit(aa.validate(
                _load_recipe(args.recipe), data=args.data, expected_outputs=args.expected_outputs,
                acceptance_criteria=_criteria_list(_load_doc(args.acceptance_criteria)), request_type=args.request_type,
            ))
        elif cmd == "smoke":
            _emit(aa.smoke(_load_recipe(args.recipe), sample=args.sample, data=args.data,
                           output_dir=args.output_dir, bootstrap_ray=args.bootstrap_ray,
                           calibration=_calibration_arg(args.calibration)))
        elif cmd == "run":
            _emit(aa.run(_load_recipe(args.recipe), confirm=args.confirm, data=args.data,
                         output_dir=args.output_dir, checkpoint_path=args.checkpoint_path,
                         bootstrap_ray=args.bootstrap_ray, smoke_token=args.smoke_token,
                         calibration=_calibration_arg(args.calibration)))
        elif cmd == "report":
            recipe = _load_recipe(args.recipe) if args.recipe else None
            _emit(aa.report(args.output, recipe=recipe, data=args.data))
        elif cmd == "verify":
            frozen = _criteria_list(_load_doc(args.frozen_criteria)) if args.frozen_criteria else None
            rec = _load_recipe(args.verify_recipe) if args.verify_recipe else None
            _emit(aa.verify(_criteria_list(_load_doc(args.criteria)) or [], evidence=_load_doc(args.evidence),
                            frozen_criteria=frozen, recipe=rec))
        elif cmd == "resolve":
            explicit = json.loads(args.explicit) if args.explicit else None
            _emit(aa.resolve(args.stage, label=args.label, use_case=args.use_case,
                             explicit=explicit, data_driven=args.data_driven))
        elif cmd == "runs":
            _emit(aa.runs(run_id=args.run_id))
        elif cmd == "continue":
            _emit(aa.plan_continuation(_load_recipe(args.recipe), args.parent_run_id, data=args.data))
        elif cmd == "calibrate":
            _emit(aa.calibrate(_load_doc(args.smoke) or {}))
        else:  # pragma: no cover - argparse enforces the choices
            return 2
    except ValueError as e:  # bad criteria / recipe / input shape -> clean JSON, not a traceback
        _emit({"error": str(e)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
