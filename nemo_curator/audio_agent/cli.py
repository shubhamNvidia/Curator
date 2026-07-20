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

    s = sub.add_parser("smoke", help="bounded run for evidence")
    s.add_argument("--recipe", required=True)
    s.add_argument("--sample", type=int, default=10)
    s.add_argument("--data")
    s.add_argument("--output-dir")
    s.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")

    r = sub.add_parser("run", help="confirm-gated full run (0 silent runs)")
    r.add_argument("--recipe", required=True)
    r.add_argument("--confirm", nargs="?", const=True, default=False,
                   help="pass the recipe config_hash (integrity) or bare --confirm")
    r.add_argument("--data")
    r.add_argument("--output-dir")
    r.add_argument("--checkpoint-path")
    r.add_argument("--bootstrap-ray", action="store_true", help="auto-start a local Ray head if none is reachable")
    r.add_argument("--smoke-token", help="smoke-evidence token from a prior smoke (required if AUDIO_AGENT_REQUIRE_SMOKE is set)")

    rp = sub.add_parser("report", help="post-hoc report from an output manifest/dir")
    rp.add_argument("--output", required=True)
    rp.add_argument("--recipe")
    rp.add_argument("--data")
    return p


def main(argv: list[str] | None = None) -> int:
    from nemo_curator import audio_agent as aa

    args = build_parser().parse_args(argv)
    cmd = args.cmd

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
        _emit(aa.validate(_load_recipe(args.recipe), data=args.data, expected_outputs=args.expected_outputs))
    elif cmd == "smoke":
        _emit(aa.smoke(_load_recipe(args.recipe), sample=args.sample, data=args.data,
                       output_dir=args.output_dir, bootstrap_ray=args.bootstrap_ray))
    elif cmd == "run":
        _emit(aa.run(_load_recipe(args.recipe), confirm=args.confirm, data=args.data,
                     output_dir=args.output_dir, checkpoint_path=args.checkpoint_path,
                     bootstrap_ray=args.bootstrap_ray, smoke_token=args.smoke_token))
    elif cmd == "report":
        recipe = _load_recipe(args.recipe) if args.recipe else None
        _emit(aa.report(args.output, recipe=recipe, data=args.data))
    else:  # pragma: no cover - argparse enforces the choices
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
