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

"""Deterministic eval harness for the audio agent (see AGENT_TEST_PLAN.md).

Asserts on the deterministic core the host planner relies on: gold-recipe
validity, unproducible roles, capability coverage, config-strategy resolution,
acceptance/honesty verification, incremental continuation, resource
calibration, planner feasibility, and enforced guardrails. It is GPU-free and
loads no models by default, so it runs in CI on CPU; the opt-in ``execute``
branch (``AUDIO_AGENT_EVAL_EXECUTE=1``) runs a tiny real smoke.

    python -m eval.audio.run_eval                       # exit 0 iff pass-rate == 1.0
    python -m eval.audio.run_eval --report eval/audio/reports/latest.json
    AUDIO_AGENT_EVAL_EXECUTE=1 python -m eval.audio.run_eval
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_QUERIES = os.path.join(_HERE, "queries.yaml")
_TAXONOMY = os.path.join(_HERE, "taxonomy.yaml")
_RECIPES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "nemo_curator", "audio_agent", "recipes"
)
_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_recipe(query: dict) -> dict | None:
    if "recipe" in query:
        return query["recipe"]
    if "recipe_ref" in query:
        return _load_yaml(os.path.join(_RECIPES_DIR, f"{query['recipe_ref']}.yaml"))
    return None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _check_safety(query: dict) -> tuple[bool | None, str]:
    """Guardrail assertions (workspace lock, redaction, smoke token, confirm gate)."""
    from nemo_curator.audio_agent import _safety

    spec = query["safety"]
    expect = query.get("expect", {})
    kind = spec.get("kind")

    if kind == "confirm_gate":
        from nemo_curator import audio_agent as aa

        # confirm defaults False -> refuses BEFORE building/executing anything.
        r = aa.run(spec["recipe"])
        ok = r.get("status") == "refused" and "confirm" in str(r.get("reason", "")).lower()
        return ok, f"confirm_gate status={r.get('status')} reason~confirm={ok}"

    if kind == "path_violations":
        old = os.environ.get("AUDIO_AGENT_WORKSPACE")
        os.environ["AUDIO_AGENT_WORKSPACE"] = spec["workspace"]
        try:
            viol = _safety.path_violations(spec["paths"])
        finally:
            if old is None:
                os.environ.pop("AUDIO_AGENT_WORKSPACE", None)
            else:
                os.environ["AUDIO_AGENT_WORKSPACE"] = old
        ok = len(viol) >= int(expect.get("violations_min", 1))
        return ok, f"path violations={viol}"

    if kind == "redact":
        red = _safety.redact(spec["obj"])
        bad = []
        for k in expect.get("redacted_keys", []):
            v = red.get(k)
            if not (isinstance(v, str) and v.startswith("<redacted")):
                bad.append(f"{k}={v!r}")
        return (not bad), (f"unredacted: {bad}" if bad else f"redacted keys={list(expect.get('redacted_keys', []))}")

    if kind == "smoke_token":
        ch = spec["config_hash"]
        tok = _safety.smoke_token(ch)
        ok = _safety.verify_smoke_token(tok, ch) and not _safety.verify_smoke_token("wrong-token", ch)
        return ok, f"smoke_token roundtrip ok={ok}"

    return False, f"unknown safety kind {kind!r}"


def _check_plan(query: dict) -> tuple[bool, str]:
    """Planner mode-selection + feasibility on a real EnvProfile."""
    from nemo_curator.audio_agent import planner
    from nemo_curator.audio_agent.contracts import EnvProfile
    from nemo_curator.audio_agent.recipe import Recipe, build_stages
    from nemo_curator.stages.audio import agent as foundation

    spec = query["plan"]
    expect = query.get("expect", {})
    stages, issues = build_stages(Recipe.from_dict(spec["recipe"]))
    if stages is None:
        return False, f"plan build failed: {[i.get('code') for i in issues]}"
    contracts = [foundation.build_contract(s) for s in stages]
    p = planner.plan(stages, contracts, EnvProfile(**spec["env"]))
    if "plan_mode" in expect and p.mode != expect["plan_mode"]:
        return False, f"plan mode={p.mode} expected={expect['plan_mode']}"
    if "plan_feasible" in expect and p.feasible != expect["plan_feasible"]:
        return False, f"plan feasible={p.feasible} expected={expect['plan_feasible']}"
    return True, f"plan mode={p.mode} feasible={p.feasible}"


def _check_execute(query: dict) -> tuple[bool | None, str]:
    """Opt-in real smoke on a tiny manifest (skipped unless AUDIO_AGENT_EVAL_EXECUTE=1)."""
    if not _truthy_env("AUDIO_AGENT_EVAL_EXECUTE"):
        return None, "skipped (set AUDIO_AGENT_EVAL_EXECUTE=1 to run)"
    from nemo_curator import audio_agent as aa

    spec = query["execute"]
    expect = query.get("expect", {})
    data = spec.get("data")
    if data and not os.path.isabs(data):
        data = os.path.join(_ROOT, data)
    rep = aa.smoke(
        spec["recipe"], sample=spec.get("sample", 2), data=data,
        bootstrap_ray=spec.get("bootstrap_ray", False),
    )
    if rep.get("status") == "refused":
        return False, f"smoke refused: {rep.get('reason')}"
    if "ran" in expect and rep.get("ran") != expect["ran"]:
        return False, f"smoke ran={rep.get('ran')} expected={expect['ran']}"
    return True, f"smoke ran={rep.get('ran')} retained={rep.get('retained')} errors={len(rep.get('errors', []))}"


def _check(query: dict) -> tuple[bool | None, str, str]:  # noqa: C901 - one linear checklist per expectation
    """Return (passed, detail, branch). ``passed`` is None when the case is skipped."""
    from nemo_curator import audio_agent as aa

    expect = query.get("expect", {})

    if "no_stage_for" in expect:
        words = [w.lower() for w in expect["no_stage_for"]]
        blob = json.dumps(aa.discover()).lower()
        hit = [w for w in words if w in blob]
        return (not hit, f"expected no stage for {words}; found matches for {hit}" if hit else "no matching stage (refuse/redirect)", "refuse")

    if "unproducible_role" in expect:
        role = expect["unproducible_role"]
        from nemo_curator.audio_agent.index import get_index

        unp = get_index().unproducible([role])
        return (role in unp, f"role {role!r} unproducible={role in unp}", "capability")

    if "safety" in query:  # guardrails: workspace lock / redaction / smoke token / confirm gate
        passed, detail = _check_safety(query)
        return passed, detail, "safety"

    if "plan" in query:  # planner mode selection + feasibility
        passed, detail = _check_plan(query)
        return passed, detail, "plan"

    if "execute" in query:  # opt-in real smoke (GPU/Ray)
        passed, detail = _check_execute(query)
        return passed, detail, "execute"

    if "verify_evidence" in query:  # acceptance verifier (1A.1/1A.3): criteria vs evidence -> report
        rep = aa.verify(query.get("acceptance_criteria", []), query["verify_evidence"],
                        frozen_criteria=query.get("frozen_criteria"))
        if "acceptance_overall" in expect and rep["overall"] != expect["acceptance_overall"]:
            return False, f"acceptance overall={rep['overall']} expected={expect['acceptance_overall']}", "verify"
        statuses = {c["id"]: c["status"] for c in rep["criteria"]}
        for cid, want in (expect.get("criterion_status") or {}).items():
            if statuses.get(cid) != want:
                return False, f"criterion {cid} status={statuses.get(cid)} expected={want}", "verify"
        if "honesty_flagged" in expect and bool(rep["honesty"]) != expect["honesty_flagged"]:
            return False, f"honesty flagged={bool(rep['honesty'])} expected={expect['honesty_flagged']}", "verify"
        if "honesty_code" in expect:
            codes = {h["code"] for h in rep["honesty"]}
            if expect["honesty_code"] not in codes:
                return False, f"honesty codes={sorted(codes)} expected {expect['honesty_code']!r}", "verify"
        return True, f"acceptance overall={rep['overall']} honesty={[h['code'] for h in rep['honesty']]}", "verify"

    if "calibrate_smoke" in query:  # 1C.2: extract measured resources from a smoke report
        cal = aa.calibrate(query["calibrate_smoke"])["calibration"]
        for stage, want in (expect.get("calibration") or {}).items():
            got = cal.get(stage, {})
            for k, v in want.items():
                if got.get(k) != v:
                    return False, f"calibration[{stage}].{k}={got.get(k)} expected {v}", "calibrate"
        return True, f"calibration={cal}", "calibrate"

    if "calibrate_plan" in query:  # 1C.2: planner prefers measured calibration over card facts
        from nemo_curator.audio_agent import planner
        from nemo_curator.audio_agent._resolve import resolve_stage_class
        from nemo_curator.audio_agent.contracts import EnvProfile
        from nemo_curator.stages.audio import agent as foundation

        spec = query["calibrate_plan"]
        st = resolve_stage_class(spec["stage"])(**(spec.get("params") or {}))
        p = planner.plan([st], [foundation.build_contract(st)], EnvProfile(**spec["env"]),
                         calibration=spec.get("calibration"))
        ps = p.per_stage[0]
        if "expect_gpu_mem_gb" in expect and ps["gpu_mem_gb"] != expect["expect_gpu_mem_gb"]:
            return False, f"gpu_mem_gb={ps['gpu_mem_gb']} expected={expect['expect_gpu_mem_gb']}", "calibrate"
        if "expect_source" in expect and ps["source"] != expect["expect_source"]:
            return False, f"source={ps['source']} expected={expect['expect_source']}", "calibrate"
        if "expect_feasible" in expect and p.feasible != expect["expect_feasible"]:
            return False, f"feasible={p.feasible} expected={expect['expect_feasible']}", "calibrate"
        return True, f"plan gpu_mem={ps['gpu_mem_gb']} source={ps['source']} feasible={p.feasible}", "calibrate"

    if "continuation" in query:  # incremental continuation (Run Records): parent + new recipe -> plan
        from nemo_curator.audio_agent import continuation as cont
        from nemo_curator.audio_agent.contracts import RunRecord
        from nemo_curator.audio_agent.recipe import Recipe

        spec = query["continuation"]
        plan = cont.plan_continuation(
            Recipe.from_dict(spec["recipe"]), RunRecord.from_dict(spec["parent"]),
            data_fingerprint=spec.get("data_fingerprint"),
        )
        if "mode" in expect and plan["mode"] != expect["mode"]:
            return False, f"continuation mode={plan['mode']} expected={expect['mode']}", "continuation"
        if "run_stages" in expect and plan.get("run_stages") != expect["run_stages"]:
            return False, f"continuation run_stages={plan.get('run_stages')} expected={expect['run_stages']}", "continuation"
        return True, f"continuation mode={plan['mode']} run={plan.get('run_stages')}", "continuation"

    if "resolve" in query:  # config-strategy resolver (1A.2): outcome -> concrete config
        spec = query["resolve"]
        r = aa.resolve(spec["stage"], label=spec.get("label"), use_case=spec.get("use_case"),
                       explicit=spec.get("explicit"))
        for param, want in (expect.get("resolve_params") or {}).items():
            if r["params"].get(param) != want:
                return False, f"resolve {param}={r['params'].get(param)} expected={want}", "resolve"
        if "resolve_filter_ref" in expect:
            got = (r["filter_stage"] or {}).get("ref")
            if got != expect["resolve_filter_ref"]:
                return False, f"resolve filter_ref={got} expected={expect['resolve_filter_ref']}", "resolve"
        if "resolve_ask" in expect and bool(r["asks"]) != expect["resolve_ask"]:
            return False, f"resolve asks={r['asks']} expected_ask={expect['resolve_ask']}", "resolve"
        return True, f"resolve params={r['params']} filter={(r['filter_stage'] or {}).get('ref')}", "resolve"

    recipe = _resolve_recipe(query)
    if recipe is None:
        return False, "query has neither recipe, recipe_ref, nor a role/capability expectation", "validate"

    v = aa.validate(
        recipe,
        expected_outputs=query.get("expected_outputs"),
        acceptance_criteria=query.get("acceptance_criteria"),
        request_type=query.get("request_type"),
    )
    if "validate_ok" in expect and v["ok"] != expect["validate_ok"]:
        return False, f"validate ok={v['ok']} expected={expect['validate_ok']}", "validate"
    if "runnable" in expect and v["runnable"] != expect["runnable"]:
        return False, f"runnable={v['runnable']} expected={expect['runnable']}", "validate"
    if "status" in expect and v.get("status") != expect["status"]:
        return False, f"status={v.get('status')} expected={expect['status']}", "validate"
    if "has_code" in expect:
        codes = {i["code"] for pool in ("issues", "card_violations", "gate_flags") for i in v[pool]}
        if expect["has_code"] not in codes:
            return False, f"expected issue code {expect['has_code']!r}; got {sorted(codes)}", "validate"
    return True, f"ok={v['ok']} runnable={v['runnable']} status={v.get('status')}", "validate"


def _card_conformance_preflight() -> tuple[bool, str]:
    """Card <-> stage conformance: fail on drift (bad params / preset / model_version)
    or orphan cards; coverage gaps (uncarded stages) are reported, not failed."""
    from nemo_curator.audio_agent.card_conformance import audit

    a = audit()
    detail = (
        f"cards {a['carded_count']}/{a['stage_count']}; drifted={len(a['violations'])} "
        f"orphan={len(a['orphan_cards'])} uncarded={len(a['uncarded_stages'])}"
    )
    if a["violations"]:
        detail += f" :: {sorted(a['violations'])}"
    return (not a["violations"] and not a["orphan_cards"]), detail


# --------------------------------------------------------------------------- #
# Reporting: per-category aggregation + a failure-report / dashboard emitter.
# --------------------------------------------------------------------------- #

def _load_taxonomy() -> dict:
    try:
        return _load_yaml(_TAXONOMY) or {}
    except Exception:  # noqa: BLE001 - reporting must never crash the run
        return {}


def _classify(detail: str, taxonomy: dict) -> tuple[str | None, str | None]:
    """Best-effort: map a failing case's detail to a (category, severity) via any
    known issue code mentioned in it."""
    codes = (taxonomy or {}).get("codes", {}) or {}
    for code, meta in codes.items():
        if code in (detail or ""):
            return meta.get("category"), meta.get("severity")
    return None, None


def _aggregate(results: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    for r in results:
        d = out.setdefault(r.get(key, ""), {"total": 0, "passed": 0, "failed": 0, "skipped": 0})
        d["total"] += 1
        if r["passed"] is True:
            d["passed"] += 1
        elif r["passed"] is False:
            d["failed"] += 1
        else:
            d["skipped"] += 1
    return out


def _build_dashboard(results: list[dict]) -> dict:
    taxonomy = _load_taxonomy()
    n_pass = sum(1 for r in results if r["passed"] is True)
    n_fail = sum(1 for r in results if r["passed"] is False)
    n_skip = sum(1 for r in results if r["passed"] is None)
    graded = n_pass + n_fail

    failures = []
    by_severity: dict[str, int] = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    for r in results:
        if r["passed"] is False:
            cat, sev = _classify(r["detail"], taxonomy)
            if sev in by_severity:
                by_severity[sev] += 1
            failures.append({
                "case_id": r["id"], "label": r["label"], "branch": r["branch"],
                "failure_category": cat, "severity": sev, "actual_behavior": r["detail"],
                "status": "open",
            })
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "totals": {
            "executed": len(results), "passed": n_pass, "failed": n_fail,
            "skipped": n_skip, "pass_rate": round(n_pass / graded, 3) if graded else 0.0,
        },
        "by_category": _aggregate(results, "label"),
        "by_branch": _aggregate(results, "branch"),
        "by_severity": by_severity,
        "failures": failures,
        "skipped": [r["id"] for r in results if r["passed"] is None],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audio agent eval harness")
    ap.add_argument("--min-pass-rate", type=float, default=1.0)
    ap.add_argument("--queries", default=_QUERIES)
    ap.add_argument("--skip-card-gate", action="store_true", help="skip the card-conformance preflight")
    ap.add_argument("--report", default=None, help="write a JSON failure-report / dashboard to this path")
    args = ap.parse_args(argv)

    queries = _load_yaml(args.queries)["queries"]
    results: list[dict] = []

    if not args.skip_card_gate:
        try:
            passed, detail = _card_conformance_preflight()
        except Exception as e:  # noqa: BLE001 - a harness error is a failed case, not a crash
            passed, detail = False, f"harness error: {type(e).__name__}: {e}"
        results.append({"id": "card_conformance", "label": "gate", "branch": "gate", "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {'card_conformance':<28} {detail}")

    for q in queries:
        try:
            passed, detail, branch = _check(q)
        except Exception as e:  # noqa: BLE001 - a harness error is a failed case, not a crash
            passed, detail, branch = False, f"harness error: {type(e).__name__}: {e}", "error"
        results.append({"id": q["id"], "label": q.get("label", ""), "branch": branch, "passed": passed, "detail": detail})
        tag = "PASS" if passed is True else ("SKIP" if passed is None else "FAIL")
        print(f"[{tag}] {q['id']:<28} {detail}")

    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"] is True)
    n_fail = sum(1 for r in results if r["passed"] is False)
    n_skip = sum(1 for r in results if r["passed"] is None)
    graded = n_pass + n_fail
    rate = n_pass / graded if graded else 0.0

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(_build_dashboard(results), f, indent=2)
        print(f"[report] wrote {args.report}")

    print(json.dumps({"total": n_total, "passed": n_pass, "failed": n_fail, "skipped": n_skip, "pass_rate": round(rate, 3)}))
    return 0 if rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
