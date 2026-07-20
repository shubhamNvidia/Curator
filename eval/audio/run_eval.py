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

"""GPU-free eval harness for the audio agent.

Asserts on the deterministic oracle the host planner relies on: gold-recipe
validity, unproducible roles, and capability coverage. No models are loaded and
no pipelines are executed, so it runs in CI on CPU. Doubles as the A/B vehicle
(contract vs skill) since it reports a validity rate.

    python -m eval.audio.run_eval           # exit 0 iff all queries pass
    python eval/audio/run_eval.py --min-pass-rate 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_QUERIES = os.path.join(_HERE, "queries.yaml")
_RECIPES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "nemo_curator", "audio_agent", "recipes"
)


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


def _check(query: dict) -> tuple[bool, str]:  # noqa: C901 - one linear checklist per expectation
    from nemo_curator import audio_agent as aa

    expect = query.get("expect", {})

    if "no_stage_for" in expect:
        words = [w.lower() for w in expect["no_stage_for"]]
        blob = json.dumps(aa.discover()).lower()
        hit = [w for w in words if w in blob]
        return (not hit, f"expected no stage for {words}; found matches for {hit}" if hit else "no matching stage (refuse/redirect)")

    if "unproducible_role" in expect:
        role = expect["unproducible_role"]
        from nemo_curator.audio_agent.index import get_index

        unp = get_index().unproducible([role])
        return (role in unp, f"role {role!r} unproducible={role in unp}")

    if "verify_evidence" in query:  # acceptance verifier (1A.1): criteria vs evidence -> report
        rep = aa.verify(query.get("acceptance_criteria", []), query["verify_evidence"])
        if "acceptance_overall" in expect and rep["overall"] != expect["acceptance_overall"]:
            return False, f"acceptance overall={rep['overall']} expected={expect['acceptance_overall']}"
        statuses = {c["id"]: c["status"] for c in rep["criteria"]}
        for cid, want in (expect.get("criterion_status") or {}).items():
            if statuses.get(cid) != want:
                return False, f"criterion {cid} status={statuses.get(cid)} expected={want}"
        return True, f"acceptance overall={rep['overall']}"

    if "resolve" in query:  # config-strategy resolver (1A.2): outcome -> concrete config
        spec = query["resolve"]
        r = aa.resolve(spec["stage"], label=spec.get("label"), use_case=spec.get("use_case"),
                       explicit=spec.get("explicit"))
        for param, want in (expect.get("resolve_params") or {}).items():
            if r["params"].get(param) != want:
                return False, f"resolve {param}={r['params'].get(param)} expected={want}"
        if "resolve_filter_ref" in expect:
            got = (r["filter_stage"] or {}).get("ref")
            if got != expect["resolve_filter_ref"]:
                return False, f"resolve filter_ref={got} expected={expect['resolve_filter_ref']}"
        if "resolve_ask" in expect and bool(r["asks"]) != expect["resolve_ask"]:
            return False, f"resolve asks={r['asks']} expected_ask={expect['resolve_ask']}"
        return True, f"resolve params={r['params']} filter={(r['filter_stage'] or {}).get('ref')}"

    recipe = _resolve_recipe(query)
    if recipe is None:
        return False, "query has neither recipe, recipe_ref, nor a role/capability expectation"

    v = aa.validate(
        recipe,
        expected_outputs=query.get("expected_outputs"),
        acceptance_criteria=query.get("acceptance_criteria"),
        request_type=query.get("request_type"),
    )
    if "validate_ok" in expect and v["ok"] != expect["validate_ok"]:
        return False, f"validate ok={v['ok']} expected={expect['validate_ok']}"
    if "runnable" in expect and v["runnable"] != expect["runnable"]:
        return False, f"runnable={v['runnable']} expected={expect['runnable']}"
    if "status" in expect and v.get("status") != expect["status"]:
        return False, f"status={v.get('status')} expected={expect['status']}"
    if "has_code" in expect:
        codes = {i["code"] for pool in ("issues", "card_violations", "gate_flags") for i in v[pool]}
        if expect["has_code"] not in codes:
            return False, f"expected issue code {expect['has_code']!r}; got {sorted(codes)}"
    return True, f"ok={v['ok']} runnable={v['runnable']}"


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audio agent eval harness")
    ap.add_argument("--min-pass-rate", type=float, default=1.0)
    ap.add_argument("--queries", default=_QUERIES)
    ap.add_argument("--skip-card-gate", action="store_true", help="skip the card-conformance preflight")
    args = ap.parse_args(argv)

    queries = _load_yaml(args.queries)["queries"]
    results = []

    if not args.skip_card_gate:
        try:
            passed, detail = _card_conformance_preflight()
        except Exception as e:  # noqa: BLE001 - a harness error is a failed case, not a crash
            passed, detail = False, f"harness error: {type(e).__name__}: {e}"
        results.append({"id": "card_conformance", "label": "gate", "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {'card_conformance':<26} {detail}")

    for q in queries:
        try:
            passed, detail = _check(q)
        except Exception as e:  # noqa: BLE001 - a harness error is a failed case, not a crash
            passed, detail = False, f"harness error: {type(e).__name__}: {e}"
        results.append({"id": q["id"], "label": q.get("label", ""), "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {q['id']:<26} {detail}")

    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    rate = n_pass / n if n else 0.0
    print(json.dumps({"total": n, "passed": n_pass, "pass_rate": round(rate, 3)}))
    return 0 if rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
