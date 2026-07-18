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

from nemo_curator.audio_agent import context as _context
from nemo_curator.audio_agent.contracts import Issue, SmokeReport, Verdict
from nemo_curator.audio_agent.index import get_index
from nemo_curator.audio_agent.profiler import probe_env, profile_data
from nemo_curator.audio_agent.recipe import Recipe, build_stages
from nemo_curator.audio_agent.report import build_run_report

# Keys in a recipe's source stage that name the input dataset (for smoke bounding).
_SOURCE_INPUT_KEYS = ("manifest_path", "input_manifest", "manifest", "raw_data_dir", "file_paths")
_MAX_SPEAKERS_KEYS = ("max_speakers", "num_speakers")


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
) -> dict[str, Any]:
    """Validate a recipe: does it compose, and can it run in this environment?

    Structural (IR + known stages) + semantic (role/key composition via the
    foundation ``validate_pipeline``) + card constraints + pre-flight gates.
    """
    from nemo_curator.stages.audio import agent as foundation

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

    available_gpus = float(env.gpu_count) if env.has_gpu else 0.0
    report = foundation.validate_pipeline(
        stages, initial_roles=roles0, initial_keys=keys0, available_gpus=available_gpus
    )
    verdict.ok = report.ok
    verdict.keys_ok = report.keys_ok
    verdict.produced_roles = sorted(report.produced_roles)
    verdict.produced_keys = sorted(report.produced_keys)
    for pi in report.issues:
        verdict.issues.append(
            Issue(pi.code, pi.severity, pi.message, stage_index=pi.stage_index, stage=pi.stage_name, fix=_fix_for(pi.code))
        )

    verdict.card_violations.extend(_card_constraint_issues(rec, data_profile))
    verdict.gate_flags.extend(_gate_issues(stages, env))
    verdict.unproducible_roles = _unproducible_roles(stages)
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


def _fix_for(code: str) -> str | None:
    return {
        "unsatisfied_reads": "insert an upstream stage that produces the missing role (see find_producers)",
        "dangling_key": "align the producer's *_key value with what this stage reads, or seed it from the source manifest",
        "tensor_into_sink": "route through AudioToDocumentStage before the JSON writer",
        "gpu_unavailable": "run on a GPU host or set the stage to CPU resources",
        "composite": "decompose the composite (it hides its true I/O) before validating downstream",
    }.get(code)


def _card_constraint_issues(recipe: Recipe, data_profile: dict[str, Any] | None) -> list[Issue]:
    """Check model-card constraints (batch, sample-rate, duration) that source can't reveal."""
    idx = get_index()
    out: list[Issue] = []
    data_srs = set((data_profile or {}).get("sample_rates", {}).keys()) if data_profile else set()
    mean_dur = float((data_profile or {}).get("mean_duration_sec", 0.0)) if data_profile else 0.0
    for i, s in enumerate(recipe.stages):
        card = idx.card(s.ref)
        if not card:
            continue
        cons = card.get("constraints", {}) or {}
        bs = cons.get("batch_size")
        if isinstance(bs, dict) and "fixed" in bs and s.params.get("batch_size") not in (None, bs["fixed"]):
            out.append(
                Issue(
                    "card_batch_size", "error",
                    f"{s.ref}: batch_size must be {bs['fixed']} ({bs.get('reason', 'model constraint')})",
                    stage_index=i, stage=s.ref, fix=f"set batch_size={bs['fixed']}",
                )
            )
        supported = cons.get("supported_sample_rates")
        if supported and data_srs and not data_srs.issubset(set(supported)):
            out.append(
                Issue(
                    "card_sample_rate", "warning",
                    f"{s.ref}: input sample rates {sorted(data_srs)} not all in supported {supported}",
                    stage_index=i, stage=s.ref, fix="insert a resample/mono stage upstream to the supported rate",
                )
            )
        sweet = cons.get("input_duration_sweetspot_sec")
        if isinstance(sweet, dict) and sweet.get("max") and mean_dur and mean_dur > float(sweet["max"]):
            out.append(
                Issue(
                    "card_duration", "warning",
                    f"{s.ref}: mean input duration {mean_dur}s exceeds sweet-spot max {sweet['max']}s",
                    stage_index=i, stage=s.ref, fix="segment/split long audio upstream",
                )
            )
        for key in _MAX_SPEAKERS_KEYS:
            mx = cons.get("max_speakers")
            if mx and s.params.get(key) and int(s.params[key]) > int(mx):
                out.append(
                    Issue(
                        "card_max_speakers", "error",
                        f"{s.ref}: {key}={s.params[key]} exceeds model max_speakers={mx}",
                        stage_index=i, stage=s.ref, fix=f"set {key}<={mx}",
                    )
                )
    return out


def _gate_issues(stages: list[Any], env: Any) -> list[Issue]:  # noqa: ANN401
    from nemo_curator.stages.audio import agent as foundation

    out: list[Issue] = []
    for idx, st in enumerate(stages):
        try:
            gates = foundation.build_contract(st).gates
        except Exception:  # noqa: BLE001
            continue
        name = type(st).__name__
        if getattr(gates, "requires_ffmpeg", False) and not env.has_ffmpeg:
            out.append(Issue("ffmpeg_missing", "error", f"{name} needs ffmpeg but it is not on PATH", stage_index=idx, stage=name, fix="install ffmpeg"))
        if getattr(gates, "requires_gpu", False) and not env.has_gpu:
            out.append(Issue("gpu_unavailable", "warning", f"{name} declares requires_gpu but no GPU was detected", stage_index=idx, stage=name, fix="run on a GPU host or lower resources"))
        if getattr(gates, "requires_internet_first_run", False):
            out.append(Issue("internet_first_run", "info", f"{name} downloads a model on first run", stage_index=idx, stage=name))
        for secret in getattr(gates, "runtime_secrets", []) or []:
            if secret not in env.available_secrets:
                out.append(Issue("missing_secret", "warning", f"{name} needs secret {secret!r} which is not set", stage_index=idx, stage=name, fix=f"export {secret}"))
    return out


def _unproducible_roles(stages: list[Any]) -> list[str]:  # noqa: ANN401
    from nemo_curator.stages.audio import agent as foundation

    required: set[str] = set()
    for st in stages:
        try:
            c = foundation.build_contract(st)
        except Exception:  # noqa: BLE001
            continue
        for key in [*c.reads.data_keys, *c.reads.segment_data_keys]:
            required.add(c.key_roles.get(key, "unknown"))
    return get_index().unproducible(sorted(required - {"unknown"}))


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

    rec = _as_recipe(recipe)
    rpt = SmokeReport(sample=sample)
    bounded, tmp_paths = _bound_recipe(rec, sample, rpt)

    stages, issues = build_stages(bounded)
    if stages is None:
        rpt.errors.extend(i.get("message", "") for i in issues)
        _cleanup(tmp_paths)
        return rpt.to_dict()

    data_profile = profile_data(data).to_dict() if data else None
    rpt.input_count = int((data_profile or {}).get("num_files", 0)) or sample
    t0 = time.perf_counter()
    try:
        if bootstrap_ray and executor is None:
            rpt.notes.append("ray_head=" + _bootstrap_ray())
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
    return rpt.to_dict()


def run(
    recipe: Recipe | dict[str, Any],
    *,
    confirm: bool | str = False,
    data: str | None = None,
    executor: Any = None,  # noqa: ANN401
    output_dir: str | None = None,
    checkpoint_path: str | None = None,
    bootstrap_ray: bool = False,
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
    env = probe_env().to_dict()

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

    stages, issues = build_stages(rec)
    if stages is None:
        return {"status": "error", "recipe_id": rec.recipe_id, "issues": issues}

    failures: list[dict[str, Any]] = []
    results: list[Any] | None = None
    t0 = time.perf_counter()
    try:
        if bootstrap_ray and executor is None:
            _bootstrap_ray()
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
    return {"status": "completed" if not failures else "failed", "report": report_obj.to_dict()}


def report(output: str, *, recipe: Recipe | dict[str, Any] | None = None, data: str | None = None) -> dict[str, Any]:
    """Post-hoc report from an output manifest/dir (counts rows vs input scale)."""
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
    return d


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
