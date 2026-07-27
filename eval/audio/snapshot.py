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

"""Regression snapshots for the audio agent (AGENT_TEST_PLAN.md 10).

Pins the deterministic surface that must not silently change:
  * `config_hash` of every library recipe (Recipe IR stability),
  * `resolve` outputs for the metrics cards (outcome -> parameter stability),
  * `validate` verdict `status` for a canonical set of recipes.

    python -m eval.audio.snapshot --write    # (re)baseline
    python -m eval.audio.snapshot --check     # exit 1 if anything drifted
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_RECIPES_DIR = os.path.join(_ROOT, "nemo_curator", "audio_agent", "recipes")
_BASELINE = os.path.join(_HERE, "reports", "snapshots", "baseline.json")

# Canonical recipes whose verdict status must stay stable.
_CANONICAL = {
    "canon_duration": {"stages": [{"ref": "GetAudioDurationStage", "params": {}}]},
    "canon_task_type_mismatch": {"stages": [
        {"ref": "ManifestReader", "params": {"manifest_path": "REQUIRED"}},
        {"ref": "AudioToDocumentStage", "params": {}},
        {"ref": "GetAudioDurationStage", "params": {}},
    ]},
    "canon_uncertain_after_composite": {"stages": [
        {"ref": "ManifestReader", "params": {"manifest_path": "REQUIRED"}},
        {"ref": "InverseTextNormalizationStage", "params": {}},
    ]},
}

_RESOLVE = [
    {"stage": "UTMOSFilterStage", "label": "studio"},
    {"stage": "BandFilterStage", "label": "wideband"},
    {"stage": "GetPairwiseWerStage", "label": "transcription_grade"},
]


def compute() -> dict:
    from nemo_curator import audio_agent as aa
    from nemo_curator.audio_agent.recipe import Recipe

    recipes = {}
    for f in sorted(glob.glob(os.path.join(_RECIPES_DIR, "*.yaml"))):
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f, encoding="utf-8") as fh:
            import yaml

            rec = Recipe.from_dict(yaml.safe_load(fh)).freeze()
        recipes[name] = rec.config_hash

    resolve = {}
    for spec in _RESOLVE:
        r = aa.resolve(spec["stage"], label=spec.get("label"))
        key = f"{spec['stage']}:{spec.get('label')}"
        resolve[key] = {"params": r.get("params"), "filter": (r.get("filter_stage") or {}).get("ref")}

    verdicts = {name: aa.validate(rec).get("status") for name, rec in _CANONICAL.items()}
    return {"recipes": recipes, "resolve": resolve, "verdicts": verdicts}


def _diff(base: dict, cur: dict) -> list[str]:
    out: list[str] = []
    for section in ("recipes", "resolve", "verdicts"):
        b, c = base.get(section, {}), cur.get(section, {})
        for k in sorted(set(b) | set(c)):
            if b.get(k) != c.get(k):
                out.append(f"{section}.{k}: baseline={b.get(k)!r} current={c.get(k)!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audio agent regression snapshots")
    ap.add_argument("--write", action="store_true", help="write/update the baseline")
    ap.add_argument("--check", action="store_true", help="diff current vs baseline (exit 1 on drift)")
    args = ap.parse_args(argv)

    cur = compute()
    if args.write or not os.path.exists(_BASELINE):
        os.makedirs(os.path.dirname(_BASELINE), exist_ok=True)
        with open(_BASELINE, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)
        print(f"[snapshot] wrote baseline -> {_BASELINE}")
        return 0

    if args.check:
        with open(_BASELINE, encoding="utf-8") as f:
            base = json.load(f)
        diffs = _diff(base, cur)
        if diffs:
            print("[snapshot] DRIFT DETECTED:")
            for d in diffs:
                print("  -", d)
            return 1
        print("[snapshot] OK (no drift vs baseline)")
        return 0

    print(json.dumps(cur, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
