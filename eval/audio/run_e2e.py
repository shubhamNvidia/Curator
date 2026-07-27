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

"""GPU/real-data E2E suite (AGENT_TEST_PLAN.md, L-E2E).

Reuses the deep-test FLEURS assets in /tmp/aa_real: for each recipe it runs the
real smoke -> run -> verify path (Ray on :6457, models already cached) and records
execution success + key metrics into reports/e2e.json.

    export RAY_ADDRESS=127.0.0.1:6457 AUDIO_AGENT_WORKSPACE=/tmp/aa_real
    python -m eval.audio.run_e2e --sample 4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPORT = os.path.join(_HERE, "reports", "e2e.json")
_AA_REAL = os.environ.get("AUDIO_AGENT_WORKSPACE", "/tmp/aa_real")
_MANIFEST = os.path.join(_AA_REAL, "manifest.jsonl")
_RECIPES = ["recipe_quality", "recipe_asr", "recipe_computewer", "recipe_vad", "recipe_diar", "recipe_ensemble"]

_YIELD_CRIT = [{"id": "kept", "type": "yield", "kind": "absolute", "severity": "must",
                "check": {"op": ">", "value": 0}}]


def _metrics_summary(report: dict) -> dict:
    """Compact per-stage metric means from a run report's examples (if present)."""
    out: dict[str, float] = {}
    examples = report.get("examples") or []
    agg: dict[str, list] = {}
    for ex in examples:
        m = ex.get("metrics") if isinstance(ex, dict) else None
        if isinstance(m, dict):
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(float(v))
    for k, vals in agg.items():
        out[k] = round(sum(vals) / len(vals), 3)
    return out


def run_one(name: str, *, sample: int) -> dict:
    import yaml

    from nemo_curator import audio_agent as aa

    path = os.path.join(_AA_REAL, f"{name}.yaml")
    if not os.path.exists(path):
        return {"recipe": name, "status": "missing", "error": f"{path} not found"}
    recipe = yaml.safe_load(open(path, encoding="utf-8"))
    row: dict = {"recipe": name}
    try:
        sm = aa.smoke(recipe, sample=sample, data=_MANIFEST)
        if sm.get("status") == "refused":
            return {**row, "status": "smoke_refused", "reason": sm.get("reason")}
        row["smoke"] = {"ran": sm.get("ran"), "retained": sm.get("retained"),
                        "rejected": sm.get("rejected"), "errors": len(sm.get("errors") or [])}
        ch, tok = sm.get("config_hash"), sm.get("smoke_token")
        rr = aa.run(recipe, confirm=ch, data=_MANIFEST, smoke_token=tok)
        row["status"] = rr.get("status")
        rep = rr.get("report") or {}
        row["accepted"] = rep.get("accepted")
        row["input_count"] = rep.get("input_count")
        row["failures"] = [f.get("code") for f in (rep.get("failure_reasons") or [])]
        row["metrics"] = _metrics_summary(rep)
        row["mode"] = ((recipe.get("machine_plan") or {}) or (rr.get("machine_plan") or {})).get("mode")
        ev = {"retained": rep.get("accepted") or 0, "input_count": rep.get("input_count") or 0}
        vr = aa.verify(_YIELD_CRIT, ev)
        row["verify_overall"] = vr.get("overall")
    except Exception as e:  # noqa: BLE001 - one recipe failing must not abort the suite
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GPU/real-data E2E suite")
    ap.add_argument("--sample", type=int, default=4, help="smoke sample size")
    ap.add_argument("--recipes", default=None, help="comma-separated subset of recipe basenames")
    ap.add_argument("--report", default=_DEFAULT_REPORT)
    args = ap.parse_args(argv)

    names = [r.strip() for r in args.recipes.split(",")] if args.recipes else _RECIPES
    rows = []
    for name in names:
        print(f"[e2e] running {name} ...", flush=True)
        row = run_one(name, sample=args.sample)
        rows.append(row)
        print(f"[e2e] {name}: status={row.get('status')} accepted={row.get('accepted')}/"
              f"{row.get('input_count')} verify={row.get('verify_overall')} metrics={row.get('metrics')}")

    completed = sum(1 for r in rows if r.get("status") == "completed")
    rep = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "workspace": _AA_REAL, "manifest": _MANIFEST,
        "totals": {"recipes": len(rows), "completed": completed,
                   "execution_success_rate": round(completed / len(rows), 3) if rows else 0.0},
        "recipes": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep["totals"]))
    print(f"[e2e] wrote {args.report}")
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
