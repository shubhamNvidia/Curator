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

    recipe = _resolve_recipe(query)
    if recipe is None:
        return False, "query has neither recipe, recipe_ref, nor a role/capability expectation"

    v = aa.validate(recipe)
    if "validate_ok" in expect and v["ok"] != expect["validate_ok"]:
        return False, f"validate ok={v['ok']} expected={expect['validate_ok']}"
    if "runnable" in expect and v["runnable"] != expect["runnable"]:
        return False, f"runnable={v['runnable']} expected={expect['runnable']}"
    if "has_code" in expect:
        codes = {i["code"] for pool in ("issues", "card_violations", "gate_flags") for i in v[pool]}
        if expect["has_code"] not in codes:
            return False, f"expected issue code {expect['has_code']!r}; got {sorted(codes)}"
    return True, f"ok={v['ok']} runnable={v['runnable']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audio agent eval harness")
    ap.add_argument("--min-pass-rate", type=float, default=1.0)
    ap.add_argument("--queries", default=_QUERIES)
    args = ap.parse_args(argv)

    queries = _load_yaml(args.queries)["queries"]
    results = []
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
