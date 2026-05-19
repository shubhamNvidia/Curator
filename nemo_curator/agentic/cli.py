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
"""``curator-adv`` — agentic-layer CLI.

Phase 1 scope (this file):

- ``curator-adv list-stages``
- ``curator-adv inspect <StageName>``
- ``curator-adv lint`` (drift between cards and class signatures)

Phase 2 extends with: ``plan``, ``run``, ``replay``, ``refine``, ``gc``.
Phase 3 swaps ``plan`` / ``run`` to use NAT under the hood.

The CLI deliberately uses argparse (already on stdlib) rather than click or
typer to avoid adding a runtime dependency for what is mostly a thin
shell-out to library code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Honor the OTel SIGSEGV workaround the audio tutorials require. We set these
# only if not already in the user's env; the user can re-enable telemetry by
# exporting OTEL_SDK_DISABLED=false before invoking the CLI.
_OTEL_DEFAULTS = {
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
}
for _k, _v in _OTEL_DEFAULTS.items():
    os.environ.setdefault(_k, _v)


# ----------------------------------------------------------------------------
# Lazy registry import
# ----------------------------------------------------------------------------


def _registry(*, strict: bool = False, cross_check: bool = False, eager: bool = False):
    """Build the registry lazily so ``--help`` doesn't pay the cost.

    Class resolution is **lazy by default** — set ``eager=True`` only when the
    command needs to import every target up front (e.g. ``lint`` and
    eventually full ``run``).
    """

    from nemo_curator.agentic.registry import build_registry  # noqa: PLC0415

    return build_registry(strict=strict, cross_check_runtime=cross_check, eager=eager)


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------


def _pretty(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, indent=2, default=str, sort_keys=False)


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------


def cmd_list_stages(args: argparse.Namespace) -> int:
    reg = _registry(cross_check=False)
    entries = sorted(reg.by_name.values(), key=lambda e: (e.card.category.value, e.card.name))

    if args.json:
        print(_pretty([e.card.model_dump(mode="json") for e in entries]))
        return 0

    # Plain table
    name_w = max((len(e.card.name) for e in entries), default=4)
    cat_w = max((len(e.card.category.value) for e in entries), default=8)
    print(f"{'NAME':<{name_w}}  {'CATEGORY':<{cat_w}}  CARDINALITY  CAPABILITIES")
    print("-" * (name_w + cat_w + 14 + 30))
    for e in entries:
        caps = ",".join(t.value for t in e.card.capabilities) or "-"
        print(
            f"{e.card.name:<{name_w}}  "
            f"{e.card.category.value:<{cat_w}}  "
            f"{e.card.produces_cardinality.value:<11}  "
            f"{caps}"
        )
    print()
    print(f"Total: {len(entries)} cards loaded")
    if reg.unresolved:
        print(f"Unresolved: {len(reg.unresolved)} (run `curator-adv inspect <name>` for details)")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    reg = _registry(cross_check=False)
    entry = reg.get(args.name)
    if entry is None:
        unresolved = [u for u in reg.unresolved if u.card.name == args.name]
        if unresolved:
            print(f"Stage {args.name!r} card found but unresolvable:")
            for u in unresolved:
                print(f"  - {u.source_path} -> {u.error}")
            return 2
        print(f"Stage {args.name!r} not found.", file=sys.stderr)
        print("Run `curator-adv list-stages` to see what is loaded.", file=sys.stderr)
        return 2

    if args.json:
        print(_pretty(entry.card))
        return 0

    c = entry.card
    print(f"# {c.name}")
    print(f"target:       {c.target}")
    print(f"category:     {c.category.value}")
    print(f"cardinality:  {c.produces_cardinality.value}")
    print(f"capabilities: {', '.join(t.value for t in c.capabilities) or '-'}")
    if c.also_handles:
        print(f"also handles: {', '.join(t.value for t in c.also_handles)}")
    print(f"resources:    cpus={c.resources.cpus} gpus={c.resources.gpus} gpu_memory_gb={c.resources.gpu_memory_gb}")
    print(f"license:      {c.license.value}  commercial_safe={c.commercial_safe}")
    print(f"cost_hint:    {c.cost_hint}")
    print(f"source:       {c.source_file}")
    print()
    print(f"summary:      {c.summary}")
    print()
    if c.description:
        for line in c.description.splitlines():
            print(line)
        print()
    inputs = c.inputs.as_tuple()
    outputs = c.outputs.as_tuple()
    print(f"inputs:       top_level={inputs[0]} data={inputs[1]}")
    print(f"outputs:      top_level={outputs[0]} data={outputs[1]}")
    if c.params:
        print()
        print("parameters:")
        for p in c.params:
            req = " (required)" if p.required else ""
            rng = ""
            if p.min is not None or p.max is not None:
                rng = f" [min={p.min}, max={p.max}]"
            ch = f" choices={p.choices}" if p.choices else ""
            print(f"  - {p.name}: {p.type}{req} default={p.default!r}{rng}{ch}")
            if p.description:
                print(f"      {p.description}")
    if c.models:
        print()
        print("models:")
        for m in c.models:
            print(f"  - {m.name} ({m.provider}, {m.license.value})")
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    reg = _registry(cross_check=False, eager=True)
    from nemo_curator.agentic.registry import lint  # noqa: PLC0415

    findings = lint(reg)
    if not findings and not reg.unresolved:
        print(f"OK — {len(reg.by_name)} cards loaded, 0 drift findings.")
        return 0

    if reg.unresolved:
        print(f"{len(reg.unresolved)} unresolved card(s):")
        for u in reg.unresolved:
            print(f"  - {u.card.name}: {u.error} ({u.source_path})")
    if findings:
        print()
        print(f"{len(findings)} drift finding(s):")
        for f in findings:
            print(f"  - [{f.kind}] {f.card_name}: {f.detail}")
    return 1


def cmd_version(_args: argparse.Namespace) -> int:
    from nemo_curator.agentic import __version__  # noqa: PLC0415

    print(__version__)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Validate + compile an IR. Emits findings.yaml + compiled.yaml under the IR's target_dir/.adv/."""

    from nemo_curator.agentic.compiler import write_compiled_yaml  # noqa: PLC0415
    from nemo_curator.agentic.ir import PipelineIR  # noqa: PLC0415
    from nemo_curator.agentic.validator import validate  # noqa: PLC0415

    ir = PipelineIR.from_path(args.ir)
    reg = _registry(cross_check=False)
    report = validate(ir, reg, intent=ir.intent, mutate=not args.no_autoinsert)
    out_dir = Path(args.out or (Path(ir.sink.target_dir) / ".adv")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report.ir.write(out_dir / "ir.validated.json")
    yaml_path = write_compiled_yaml(report.ir, reg, out_dir / "compiled.yaml")

    print(f"validated IR: {out_dir / 'ir.validated.json'}")
    print(f"compiled YAML: {yaml_path}")
    print(f"fingerprint:   {report.fingerprint}")
    if report.findings:
        print()
        for f in report.findings:
            print(f"  [{f.severity.value:7s}] {f.code}: {f.detail}")
    if not report.is_ok():
        return 2
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Validate, compile, and execute the IR. Always emits a RunCard."""

    from nemo_curator.agentic.ir import PipelineIR  # noqa: PLC0415
    from nemo_curator.agentic.runner import RunOptions, run  # noqa: PLC0415

    ir = PipelineIR.from_path(args.ir)
    if args.target_dir:
        ir = ir.model_copy(update={"sink": ir.sink.model_copy(update={"target_dir": args.target_dir})})

    reg = _registry(cross_check=False)
    opts = RunOptions(
        dry_run=args.dry_run,
        enable_cache=not args.no_cache,
        max_cache_bytes=args.max_cache_bytes,
        run_id=args.run_id,
    )
    result = run(ir, reg, options=opts)
    print(f"run_id:        {result.run_card.run_id}")
    print(f"success:       {result.success}")
    print(f"target_dir:    {result.target_dir}")
    print(f"compiled_yaml: {result.compiled_yaml_path}")
    if result.findings_yaml_path:
        print(f"findings:      {result.findings_yaml_path}")
    if result.error:
        print(f"error:         {result.error}")
    return 0 if result.success else 1


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-run a previous RunCard's IR. The run_id is preserved when --keep-run-id is set."""

    from nemo_curator.agentic.ir import PipelineIR  # noqa: PLC0415
    from nemo_curator.agentic.runner import RunOptions, run  # noqa: PLC0415

    run_card_path = Path(args.run_card).expanduser().resolve()
    if run_card_path.is_dir():
        run_card_path = run_card_path / ".adv" / "run_card.yaml"
    import yaml  # noqa: PLC0415

    info = yaml.safe_load(run_card_path.read_text(encoding="utf-8"))
    ir_path = Path(info["pipeline_ir_path"])
    ir = PipelineIR.from_path(ir_path)

    reg = _registry(cross_check=False)
    opts = RunOptions(run_id=info["run_id"] if args.keep_run_id else None)
    result = run(ir, reg, options=opts)
    print(f"replay success: {result.success}")
    return 0 if result.success else 1


def cmd_refine(args: argparse.Namespace) -> int:
    """Phase 2 placeholder. The refine loop with critic-driven adjustments lands with NAT in Phase 3."""

    print("`curator-adv refine` is wired in Phase 3 (NAT integration). For now use `plan` + `run` directly.")
    return 0


_DEFAULT_NAT_WORKFLOW = Path(__file__).parent / "nat" / "workflow.yml"


def _configure_agent_logging(*, verbose: bool) -> None:
    """Route NAT / LangChain / our own loggers to stderr at a useful level.

    NAT does not auto-configure ``logging`` and the React agent emits its
    Thought / Action / Observation cycle at INFO. Without this hook the
    process is effectively silent until the very last response. ``--verbose``
    additionally enables DEBUG for the NAT internals.
    """

    import logging  # noqa: PLC0415

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", force=True)

    for noisy in ("httpx", "httpcore", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for name in ("nat", "nat.plugins.langchain", "langchain", "langchain_core",
                 "langgraph", "nemo_curator.agentic"):
        logging.getLogger(name).setLevel(level)


def cmd_plan_from_prompt(args: argparse.Namespace) -> int:
    """Phase 3: turn a natural-language prompt into a validated pipeline.

    Two planner modes:

    - ``--planner dag`` (default): the multi-agent DAG in
      :mod:`nemo_curator.agentic.planner_dag`. Four focused LLM steps
      (intent / pick / tune / critique) wrapped around the deterministic
      validator and compiler. The flow we want as of Phase 3 closeout.
    - ``--planner react``: legacy single-React-agent path through NAT.
      Kept as an escape hatch while the DAG matures.
    """

    _configure_agent_logging(verbose=args.verbose)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ADV_AGENT_OUT_DIR"] = str(out_dir)

    if args.planner == "dag":
        return _cmd_plan_from_prompt_dag(args, out_dir)
    return _cmd_plan_from_prompt_react(args, out_dir)


def _cmd_plan_from_prompt_dag(args: argparse.Namespace, out_dir: Path) -> int:
    """Run the multi-agent planner DAG (Steps 1-6 from planner_dag.py)."""

    from nemo_curator.agentic.compiler import write_compiled_yaml  # noqa: PLC0415
    from nemo_curator.agentic.llm import LLMClient  # noqa: PLC0415
    from nemo_curator.agentic.planner_dag import plan as run_dag  # noqa: PLC0415

    if not args.dataset:
        print("--dataset is required for the DAG planner.", file=sys.stderr)
        return 2

    registry = _registry(cross_check=False)
    llm = LLMClient()
    try:
        result = run_dag(
            prompt=args.prompt,
            source_uri=args.dataset,
            source_kind=args.kind,
            target_dir=str(out_dir),
            llm=llm,
            registry=registry,
            max_refine_iters=args.max_refine_iters,
            tier=args.tier,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"planner DAG failed: {exc}", file=sys.stderr)
        return 1

    # Persist artifacts -- mirror the React-mode layout so downstream tools
    # (`curator-adv run <ir>`) work either way.
    ir_path = out_dir / "ir.validated.json"
    result.ir.write(ir_path)
    yaml_path = write_compiled_yaml(result.ir, registry, out_dir / "compiled.yaml")
    findings_path = out_dir / "findings.json"
    findings_path.write_text(
        json.dumps([
            {
                "severity": f.severity.value,
                "code": f.code,
                "detail": f.detail,
                "stage_index": f.stage_index,
                "stage_name": f.stage_name,
            }
            for f in result.report.findings
        ], indent=2),
        encoding="utf-8",
    )
    critic_path = out_dir / "critic_history.json"
    critic_path.write_text(
        json.dumps([
            {
                "approved": v.approved,
                "score": v.score,
                "complaints": v.complaints,
                "patch": v.patch,
            }
            for v in result.critic_history
        ], indent=2),
        encoding="utf-8",
    )

    print(f"validated IR: {ir_path}")
    print(f"compiled YAML: {yaml_path}")
    print(f"findings:     {findings_path}")
    print(f"critic:       {critic_path}")
    print()
    print(f"iterations:   {len(result.critic_history)}")
    last = result.critic_history[-1] if result.critic_history else None
    if last is not None:
        print(f"approved:     {last.approved}  score={last.score!r}")
        if last.complaints:
            print("complaints:")
            for c in last.complaints:
                print(f"  - {c}")
    return 0 if (last is None or last.approved) else 0  # always 0; non-approval still produces artifacts


def _cmd_plan_from_prompt_react(args: argparse.Namespace, out_dir: Path) -> int:
    """Legacy NAT React-agent path. Unchanged from earlier Phase 3 versions."""

    try:
        # Force-register our 12 tools with NAT's GlobalTypeRegistry BEFORE the
        # workflow YAML is validated. This is needed because the package is
        # typically not pip-installed in this repo, so NAT's entry-point scan
        # of ``nat.components`` would otherwise miss us.
        import nemo_curator.agentic.nat  # noqa: F401,PLC0415
        from nat.runtime.loader import load_workflow  # noqa: PLC0415
    except ImportError as exc:
        print(
            "NAT is not installed in this environment. "
            "Run: uv pip install nvidia-nat nvidia-nat-langchain",
            file=sys.stderr,
        )
        print(f"  underlying error: {exc}", file=sys.stderr)
        return 3

    workflow_path = Path(args.config or _DEFAULT_NAT_WORKFLOW).expanduser().resolve()

    user_prompt = args.prompt
    if args.dataset:
        user_prompt = (
            f"{user_prompt}\n\nDataset to curate: {args.dataset} "
            f"(kind={args.kind}). Write the result to {out_dir}."
        )

    import asyncio  # noqa: PLC0415

    async def _run() -> str:
        async with load_workflow(workflow_path) as workflow:
            async with workflow.run(user_prompt) as run_ctx:
                return await run_ctx.result(to_type=str)

    try:
        answer = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        print(f"agent run failed: {exc}", file=sys.stderr)
        return 1

    transcript_path = out_dir / "agent_transcript.txt"
    transcript_path.write_text(answer, encoding="utf-8")
    print(f"agent transcript: {transcript_path}")
    print()
    print(answer)
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    from nemo_curator.agentic.cache import StageCache  # noqa: PLC0415

    target = Path(args.target_dir).expanduser().resolve()
    cache_dir = target / ".adv" / "cache"
    cache = StageCache(cache_dir, max_bytes=args.max_bytes)
    before = cache.total_bytes()
    freed = cache.gc(target_bytes=args.max_bytes)
    after = cache.total_bytes()
    print(f"cache: was {before / 1e6:.1f} MB → now {after / 1e6:.1f} MB (freed {freed / 1e6:.1f} MB)")
    return 0


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curator-adv",
        description="Agentic-layer CLI for the audio curation pipeline.",
    )
    parser.set_defaults(func=lambda _a: parser.print_help() or 0)
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list-stages", help="List every loaded stage card.")
    p_list.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    p_list.set_defaults(func=cmd_list_stages)

    p_ins = sub.add_parser("inspect", help="Show one stage card in detail.")
    p_ins.add_argument("name", help="StageCard.name (the Python class name).")
    p_ins.add_argument("--json", action="store_true", help="Emit JSON instead of human-friendly text.")
    p_ins.set_defaults(func=cmd_inspect)

    p_lint = sub.add_parser("lint", help="Detect drift between cards and their class signatures.")
    p_lint.set_defaults(func=cmd_lint)

    p_plan = sub.add_parser("plan", help="Validate + compile an IR to canonical YAML.")
    p_plan.add_argument("ir", help="Path to a pipeline IR (JSON or YAML).")
    p_plan.add_argument("--out", default=None, help="Directory for validated artifacts (default: <target_dir>/.adv/).")
    p_plan.add_argument("--no-autoinsert", action="store_true", help="Disable Mono/Resample auto-insert.")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="Validate, compile, and execute an IR.")
    p_run.add_argument("ir", help="Path to a pipeline IR (JSON or YAML).")
    p_run.add_argument("--target-dir", default=None, help="Override the IR's sink.target_dir.")
    p_run.add_argument("--dry-run", action="store_true", help="Build the pipeline and call setup/teardown; do not execute.")
    p_run.add_argument("--no-cache", action="store_true", help="Disable the per-stage cache.")
    p_run.add_argument("--max-cache-bytes", type=int, default=50 * 1024 * 1024 * 1024)
    p_run.add_argument("--run-id", default=None)
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("replay", help="Re-run a saved run.")
    p_rep.add_argument("run_card", help="Path to run_card.yaml or its containing directory.")
    p_rep.add_argument("--keep-run-id", action="store_true", help="Reuse the prior run_id.")
    p_rep.set_defaults(func=cmd_replay)

    p_ref = sub.add_parser("refine", help="(Phase 3) iterate with the critic loop. Stub today.")
    p_ref.set_defaults(func=cmd_refine)

    p_pfp = sub.add_parser(
        "plan-from-prompt",
        help="Phase 3: turn a natural-language prompt into a validated IR + compiled YAML.",
    )
    p_pfp.add_argument("prompt", help="The user's natural-language curation request.")
    p_pfp.add_argument("--dataset", default=None, help="Dataset URI (required for the dag planner).")
    p_pfp.add_argument("--kind", default="manifest", choices=["manifest", "directory"], help="Dataset kind.")
    p_pfp.add_argument("--out", default="./adv_run", help="Output directory for artifacts.")
    p_pfp.add_argument("--config", default=None, help="NAT workflow YAML (react mode only).")
    p_pfp.add_argument(
        "--planner",
        default="dag",
        choices=["dag", "react"],
        help="Planner backend. 'dag' = multi-agent (default); 'react' = legacy single-agent NAT loop.",
    )
    p_pfp.add_argument(
        "--max-refine-iters",
        type=int,
        default=2,
        help="DAG planner: cap on critic refinement iterations (default 2).",
    )
    p_pfp.add_argument(
        "--tier",
        default="synth",
        choices=["planner", "synth"],
        help="DAG planner: LLM tier to use for every step (default 'synth').",
    )
    p_pfp.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="DEBUG-level logging for the planner and any LLM/NAT internals.",
    )
    p_pfp.set_defaults(func=cmd_plan_from_prompt)

    p_gc = sub.add_parser("gc", help="LRU-evict the per-stage cache under a target_dir.")
    p_gc.add_argument("target_dir", help="Pipeline target directory.")
    p_gc.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024 * 1024)
    p_gc.set_defaults(func=cmd_gc)

    p_ver = sub.add_parser("version", help="Print the agentic-layer version.")
    p_ver.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
