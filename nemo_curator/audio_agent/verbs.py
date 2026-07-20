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

"""The deterministic verb surface the host LLM (and CLI/MCP) drives.

Each verb returns a JSON-safe dict. Retrieval/validation/execution/reporting is
ours (deterministic, grounded); interpret/route/plan/critique is the host's.

    discover / describe / catalog_tree / cards / context   -> knowledge + routing
    validate                                               -> Verdict (grounds the plan)
    smoke                                                  -> bounded evidence
    run                                                    -> confirm-gated full run + report
    report                                                 -> post-hoc evidence from outputs
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from typing import Any

from nemo_curator.audio_agent import _safety
from nemo_curator.audio_agent import context as _context
from nemo_curator.audio_agent.contracts import Issue, SmokeReport, Verdict
from nemo_curator.audio_agent.index import get_index
from nemo_curator.audio_agent.profiler import probe_env, profile_data
from nemo_curator.audio_agent.recipe import Recipe, build_stages
from nemo_curator.audio_agent.report import build_run_report

# Keys in a recipe's source stage that name the input dataset (for smoke bounding).
_SOURCE_INPUT_KEYS = ("manifest_path", "input_manifest", "manifest", "raw_data_dir", "file_paths")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_recipe(recipe: Recipe | dict[str, Any]) -> Recipe:
    return recipe if isinstance(recipe, Recipe) else Recipe.from_dict(recipe)


def _derive_initial(data_profile: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    """Seed initial roles + literal keys from the input data profile."""
    from nemo_curator.stages.audio._roles import role_for_value

    keys: set[str] = {"audio_filepath"}
    if data_profile and data_profile.get("manifest_keys"):
        keys |= set(data_profile["manifest_keys"])
    roles = {role_for_value(k) for k in keys} | {"audio_filepath"}
    roles.discard("unknown")
    return roles, keys


# --------------------------------------------------------------------------- #
# discovery + routing (L0/L1/L2)
# --------------------------------------------------------------------------- #
def discover() -> dict[str, Any]:
    """List every agent-ready audio stage with its category + one-liner."""
    idx = get_index()
    stages = [
        {"stage": name, "category": idx.category_of(name), "summary": idx.one_liner(name), "tags": idx.tags_of(name)}
        for name in idx.stage_names()
    ]
    return {"count": len(stages), "stages": stages}


def describe(name: str) -> dict[str, Any]:
    """Return the static contract (+ card, if any) for a single stage."""
    from nemo_curator.audio_agent._resolve import static_contract_for

    out: dict[str, Any] = {"stage": name, "category": get_index().category_of(name)}
    try:
        out["contract"] = static_contract_for(name).to_dict()
    except KeyError:
        return {"stage": name, "error": f"{name!r} is not a registered agent-ready audio stage"}
    except Exception as e:  # noqa: BLE001
        out["contract_error"] = f"{type(e).__name__}: {e}"
    card = get_index().card(name)
    if card:
        out["card"] = card
    return out


def catalog_tree() -> dict[str, Any]:
    """L0: the full category tree the host prunes over before drilling in."""
    return {"categories": get_index().category_tree()}


def cards(category: str | None = None, names: list[str] | None = None) -> dict[str, Any]:
    """L1 (one-liners for a category) or L2 (full cards for named finalists)."""
    idx = get_index()
    if names:
        return {"cards": idx.full_cards(names)}
    if category:
        return {"category": category, "stages": idx.card_oneliners(category)}
    msg = "cards() requires either a category (L1) or a list of names (L2)"
    raise ValueError(msg)


def context(
    goal: dict[str, Any] | None = None,
    *,
    data: str | None = None,
    stages: list[str] | None = None,
    roles: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a compact PlanningContext for the host router/planner."""
    return _context.assemble(goal, data=data, selected_stages=stages, roles=roles).to_dict()


# --------------------------------------------------------------------------- #
# validate (structural + semantic + card + preflight)
# --------------------------------------------------------------------------- #
def validate(
    recipe: Recipe | dict[str, Any],
    *,
    data: str | None = None,
    initial_keys: list[str] | None = None,
    initial_roles: list[str] | None = None,
    expected_outputs: list[str] | None = None,
    acceptance_criteria: list[dict[str, Any]] | None = None,
    request_type: str | None = None,
) -> dict[str, Any]:
    """Validate a recipe: does it compose, and can it run in this environment?

    Well-formedness is checked here (real stages, constructible params); the rest
    runs through the pluggable check registry (``audio_agent.checks``): data-flow
    (role/key/residency/serialization), card constraints, environment gates,
    unproducible roles, task-type, output-completeness, and request-type sanity.
    ``expected_outputs`` (semantic roles) enables the output-completeness check;
    ``acceptance_criteria`` (1A.1) additionally compile their output/metric fields
    into that check and drive request-type sanity via ``request_type``.
    """
    from nemo_curator.audio_agent.acceptance import expected_roles_from_criteria, parse_criteria
    from nemo_curator.audio_agent.checks import CheckContext, run_checks

    rec = _as_recipe(recipe)
    verdict = Verdict()

    if not rec.stages:
        verdict.issues.append(Issue("empty_recipe", "error", "recipe has no stages"))
        return verdict.to_dict()

    data_profile = profile_data(data).to_dict() if data else None
    env = probe_env()

    stages, build_issues = build_stages(rec)
    verdict.issues.extend(_issue_from_dict(i) for i in build_issues)
    if stages is None:
        return verdict.to_dict()

    roles0, keys0 = _derive_initial(data_profile)
    if initial_roles is not None:
        roles0 = set(initial_roles)
    if initial_keys is not None:
        keys0 = set(initial_keys)

    criteria = parse_criteria(acceptance_criteria)
    expected = set(expected_outputs or []) | set(expected_roles_from_criteria(criteria))

    ctx = CheckContext(
        recipe=rec,
        stages=stages,
        data_profile=data_profile,
        env=env,
        initial_roles=roles0,
        initial_keys=keys0,
        available_gpus=float(env.gpu_count) if env.has_gpu else 0.0,
        expected_outputs=sorted(expected),
        acceptance_criteria=criteria,
        request_type=request_type,
    )
    result = run_checks(ctx)
    verdict.ok = bool(result.ok)
    verdict.keys_ok = bool(result.keys_ok)
    verdict.produced_roles = result.produced_roles
    verdict.produced_keys = result.produced_keys
    verdict.issues.extend(result.issues)
    verdict.card_violations.extend(result.card_violations)
    verdict.gate_flags.extend(result.gate_flags)
    verdict.unproducible_roles = result.unproducible_roles
    return verdict.to_dict()


def _issue_from_dict(d: dict[str, Any]) -> Issue:
    return Issue(
        code=d.get("code", "error"),
        severity=d.get("severity", "error"),
        message=d.get("message", ""),
        stage_index=d.get("stage_index"),
        stage=d.get("stage"),
        fix=d.get("fix"),
    )


# --------------------------------------------------------------------------- #
# smoke + run + report
# --------------------------------------------------------------------------- #
def smoke(
    recipe: Recipe | dict[str, Any],
    *,
    sample: int = 10,
    data: str | None = None,
    executor: Any = None,  # noqa: ANN401
    output_dir: str | None = None,
    bootstrap_ray: bool = False,
) -> dict[str, Any]:
    """Run the recipe on a bounded sample and return structured evidence.

    ``bootstrap_ray`` opts into auto-starting a correctly-configured local Ray
    head when none is reachable (see ``_ray.ensure_cluster``).
    """
    from nemo_curator.audio_agent.failures import classify

    rec = _as_recipe(recipe).freeze()
    pviol = _safety.path_violations([data, output_dir, *_safety.recipe_path_params(rec)])
    if pviol:
        return {"status": "refused", "reason": "path(s) resolve outside the allowed workspace", "violations": pviol}
    rpt = SmokeReport(sample=sample)
    bounded, tmp_paths = _bound_recipe(rec, sample, rpt)

    stages, issues = build_stages(bounded)
    if stages is None:
        rpt.errors.extend(i.get("message", "") for i in issues)
        _cleanup(tmp_paths)
        return _safety.redact(rpt.to_dict())

    data_profile = profile_data(data).to_dict() if data else None
    rpt.input_count = int((data_profile or {}).get("num_files", 0)) or sample
    rplan = _plan_resources(stages, probe_env(), data_profile)
    rpt.notes.append(f"resource_plan_mode={rplan.mode}")
    if rplan.escalations:
        rpt.notes.append("resource_escalations=" + "; ".join(rplan.escalations))
    caller_executor = executor
    t0 = time.perf_counter()
    try:
        if bootstrap_ray and caller_executor is None:
            rpt.notes.append("ray_head=" + _bootstrap_ray())
        if caller_executor is None:
            executor = _make_executor(rplan.mode)
        results = _run_pipeline(stages, executor)
        rpt.ran = True
        rpt.retained = len(results or [])
        rpt.rejected = max(0, min(sample, rpt.input_count) - rpt.retained)
        rpt.per_stage_metrics = _stage_metrics(results)
        rpt.examples = _examples(results, limit=3)
        rpt.goals_met = rpt.retained > 0
    except Exception as e:  # noqa: BLE001 - classify any execution failure for the critic
        rpt.errors.append(f"{type(e).__name__}: {e}")
        rpt.notes.append(str(classify(f"{type(e).__name__}: {e}")))
    finally:
        rpt.notes.append(f"elapsed_sec={round(time.perf_counter() - t0, 3)}")
        _cleanup(tmp_paths)
    out = rpt.to_dict()
    out["config_hash"] = rec.config_hash
    out["smoke_token"] = _safety.smoke_token(rec.config_hash)
    return _safety.redact(out)


def run(
    recipe: Recipe | dict[str, Any],
    *,
    confirm: bool | str = False,
    data: str | None = None,
    executor: Any = None,  # noqa: ANN401
    output_dir: str | None = None,
    checkpoint_path: str | None = None,
    bootstrap_ray: bool = False,
    smoke_token: str | None = None,
) -> dict[str, Any]:
    """Confirm-gated full run. Refuses without explicit confirmation (0 silent runs).

    ``confirm`` may be ``True`` or the recipe's ``config_hash`` (integrity: what
    was approved is what runs). Returns a refusal-with-estimate until confirmed.
    ``checkpoint_path`` enables partial-run recovery (resume completed source
    partitions on a rerun) when the pipeline's stages are resumability-safe.
    ``bootstrap_ray`` opts into auto-starting a local Ray head when none is reachable.
    """
    from nemo_curator.audio_agent.failures import classify

    rec = _as_recipe(recipe).freeze()
    data_profile = profile_data(data).to_dict() if data else None
    env_obj = probe_env()
    env = env_obj.to_dict()

    if confirm is False:
        return {
            "status": "refused",
            "reason": "full run requires explicit confirmation (0 silent full-scale runs)",
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
            "estimate": _estimate(data_profile),
            "confirm_with": f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed",
        }
    if isinstance(confirm, str) and confirm != rec.config_hash:
        return {
            "status": "refused",
            "reason": "plan-execution integrity check failed: confirmed hash does not match the recipe",
            "confirmed": confirm,
            "config_hash": rec.config_hash,
        }

    pviol = _safety.path_violations([data, output_dir, checkpoint_path, *_safety.recipe_path_params(rec)])
    if pviol:
        return {"status": "refused", "reason": "path(s) resolve outside the allowed workspace", "recipe_id": rec.recipe_id, "violations": pviol}
    if _safety.require_smoke() and not _safety.verify_smoke_token(smoke_token, rec.config_hash):
        return {
            "status": "refused",
            "reason": "run requires smoke evidence (AUDIO_AGENT_REQUIRE_SMOKE is set): run smoke on this recipe and pass its 'smoke_token'",
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
        }

    stages, issues = build_stages(rec)
    if stages is None:
        return {"status": "error", "recipe_id": rec.recipe_id, "issues": issues}

    rplan = _plan_resources(stages, env_obj, data_profile)
    rec.with_machine_plan(rplan.to_dict(), machine_fingerprint=rplan.machine_fingerprint)
    if not rplan.feasible:
        return {
            "status": "refused",
            "reason": "resource plan is infeasible on this machine",
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
            "escalations": rplan.escalations,
            "machine_plan": rplan.to_dict(),
        }

    caller_executor = executor
    failures: list[dict[str, Any]] = []
    results: list[Any] | None = None
    t0 = time.perf_counter()
    try:
        if bootstrap_ray and caller_executor is None:
            _bootstrap_ray()
        if caller_executor is None:
            executor = _make_executor(rplan.mode)
        results = _run_pipeline(stages, executor, checkpoint_path=checkpoint_path)
    except Exception as e:  # noqa: BLE001 - classify + report, do not crash the caller
        failures.append(classify(f"{type(e).__name__}: {e}"))
    elapsed = time.perf_counter() - t0

    output_paths = _recipe_outputs(rec, output_dir)
    report_obj = build_run_report(
        recipe=rec,
        result_tasks=results,
        data_profile=data_profile,
        env_profile=env,
        output_paths=output_paths,
        elapsed_sec=elapsed,
        failures=failures,
        examples=_examples(results, limit=5) if results else [],
        next_action="review retained/rejected; adjust thresholds and re-run if needed"
        if not failures
        else "triage the failure_reasons and re-validate",
    )
    return _safety.redact({"status": "completed" if not failures else "failed", "report": report_obj.to_dict()})


def report(output: str, *, recipe: Recipe | dict[str, Any] | None = None, data: str | None = None) -> dict[str, Any]:
    """Post-hoc report from an output manifest/dir (counts rows vs input scale)."""
    pviol = _safety.path_violations([output, data])
    if pviol:
        return {"status": "refused", "reason": "path(s) resolve outside the allowed workspace", "violations": pviol}
    rec = _as_recipe(recipe) if recipe is not None else None
    data_profile = profile_data(data).to_dict() if data else None
    accepted = _count_output_rows(output)
    input_count = int((data_profile or {}).get("num_files", 0))
    rpt = build_run_report(
        recipe=rec or Recipe(),
        result_tasks=[],
        data_profile=data_profile,
        env_profile=probe_env().to_dict(),
        output_paths=[output],
        next_action="compare against expected retention; scale up or adjust thresholds",
    )
    d = rpt.to_dict()
    d["accepted"] = accepted
    d["input_count"] = input_count or accepted
    d["rejected"] = max(0, (input_count or accepted) - accepted)
    return _safety.redact(d)


def verify(
    acceptance_criteria: list[dict[str, Any]],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate acceptance criteria against gathered evidence -> AcceptanceReport (1A.1).

    Deterministic verifier: it runs nothing itself, it judges the ``evidence`` the
    host assembled (from ``validate`` — ``produced_roles``/``produced_keys`` — and
    from ``smoke``/``run`` — ``metrics``/``per_item``/``retained``/``input_count``,
    plus optional ``unachievable_fields``). Returns per-criterion states
    (met / not_met / unverifiable / unachievable) and an ``overall`` that is
    ``met`` iff every ``must`` criterion is met — the anti-goalpost-moving gate.
    """
    from nemo_curator.audio_agent.acceptance import parse_criteria, verify as _verify

    report_obj = _verify(parse_criteria(acceptance_criteria), evidence or {})
    return _safety.redact(report_obj.to_dict())


# --------------------------------------------------------------------------- #
# execution + bounding helpers
# --------------------------------------------------------------------------- #
def _bootstrap_ray() -> str:
    """Ensure a Ray cluster (opt-in) and return its address; sets RAY_ADDRESS."""
    from nemo_curator.audio_agent._ray import ensure_cluster

    return ensure_cluster()


def _run_pipeline(stages: list[Any], executor: Any, *, checkpoint_path: str | None = None) -> list[Any] | None:  # noqa: ANN401
    from nemo_curator.pipeline import Pipeline

    pipeline = Pipeline(name="audio_agent_run", stages=list(stages))
    return pipeline.run(executor, checkpoint_path=checkpoint_path)


def _plan_resources(stages: list[Any], env_obj: Any, data_profile: dict[str, Any] | None):  # noqa: ANN401
    """Run the deterministic resource planner over the built stages (1C.1)."""
    from nemo_curator.audio_agent import planner
    from nemo_curator.stages.audio import agent as foundation

    contracts: list[Any] = []
    for st in stages:
        try:
            contracts.append(foundation.build_contract(st))
        except Exception:  # noqa: BLE001 - a stage that can't describe itself gets conservative defaults
            contracts.append(None)
    return planner.plan(stages, contracts, env_obj, data_profile)


def _make_executor(mode: str) -> Any:  # noqa: ANN401
    """A XennaExecutor with the planned execution_mode; None -> the pipeline default."""
    try:
        from nemo_curator.backends.xenna import XennaExecutor

        return XennaExecutor({"execution_mode": mode})
    except Exception:  # noqa: BLE001 - fall back to the pipeline's default executor (streaming)
        return None


def _bound_recipe(recipe: Recipe, sample: int, rpt: SmokeReport) -> tuple[Recipe, list[str]]:
    """Return a copy of the recipe bounded to ``sample`` items, + temp paths to clean up.

    If the source stage reads a JSONL manifest, truncate it to ``sample`` lines in
    a temp file and point the recipe at it. If the source supports ``max_samples``,
    set it. Otherwise, run unbounded with a note.
    """
    import copy

    bounded = copy.deepcopy(recipe)
    tmp_paths: list[str] = []
    if not bounded.stages:
        return bounded, tmp_paths
    src = bounded.stages[0]

    for key in _SOURCE_INPUT_KEYS:
        val = src.params.get(key)
        if isinstance(val, str) and val.endswith((".jsonl", ".json")) and os.path.isfile(os.path.expanduser(val)):
            tmp = _truncate_manifest(os.path.expanduser(val), sample)
            src.params[key] = tmp
            tmp_paths.append(tmp)
            rpt.notes.append(f"bounded via truncated manifest ({sample} lines)")
            return bounded, tmp_paths

    # source exposes a sample cap
    if "max_samples" in src.params or _accepts_param(src.ref, "max_samples"):
        src.params["max_samples"] = sample
        rpt.notes.append(f"bounded via max_samples={sample}")
        return bounded, tmp_paths

    rpt.notes.append("could not bound input; smoke ran unbounded (add max_samples or use a manifest source)")
    return bounded, tmp_paths


def _truncate_manifest(path: str, n: int) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="audio_agent_smoke_")
    with os.fdopen(fd, "w", encoding="utf-8") as out, open(path, encoding="utf-8") as src:
        written = 0
        for line in src:
            if line.strip():
                out.write(line if line.endswith("\n") else line + "\n")
                written += 1
                if written >= n:
                    break
    return tmp


def _accepts_param(ref: str, param: str) -> bool:
    import inspect

    with contextlib.suppress(Exception):
        from nemo_curator.audio_agent._resolve import resolve_stage_class

        return param in inspect.signature(resolve_stage_class(ref).__init__).parameters
    return False


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            os.remove(p)


def _estimate(data_profile: dict[str, Any] | None) -> dict[str, Any]:
    dp = data_profile or {}
    return {
        "input_count": dp.get("num_files", 0),
        "total_duration_sec": dp.get("total_duration_sec", 0.0),
        "note": "precise $/GPU-hour costing is deferred; time estimate comes from a prior smoke run",
    }


def _recipe_outputs(recipe: Recipe, output_dir: str | None) -> list[str]:
    outs: list[str] = []
    for s in recipe.stages:
        for key in ("output_path", "path", "output_manifest"):
            v = s.params.get(key)
            if isinstance(v, str):
                outs.append(v)
    if output_dir:
        outs.append(output_dir)
    return outs


def _stage_metrics(results: list[Any] | None) -> dict[str, Any]:
    from nemo_curator.audio_agent.report import _dedup_stage_perf

    return _dedup_stage_perf(results or [])


def _examples(results: list[Any] | None, *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in (results or [])[:limit]:
        data = getattr(task, "data", None)
        if isinstance(data, dict):
            out.append({k: v for k, v in data.items() if _jsonable(v)})
    return out


def _jsonable(v: Any) -> bool:  # noqa: ANN401
    return isinstance(v, (str, int, float, bool, type(None)))


def _count_output_rows(output: str) -> int:
    expanded = os.path.expanduser(output)
    files: list[str] = []
    if os.path.isdir(expanded):
        for root, _d, fs in os.walk(expanded):
            files.extend(os.path.join(root, f) for f in fs if f.endswith((".jsonl", ".json")))
    elif os.path.isfile(expanded):
        files = [expanded]
    total = 0
    for f in files:
        with contextlib.suppress(OSError), open(f, encoding="utf-8") as fh:
            total += sum(1 for line in fh if line.strip())
    return total
