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

"""Card <-> stage conformance gate: keeps capability cards honest (2.1).

Mechanical facts in a card are checked against the real stage so a card cannot
drift (the exact ``resample`` / ``audio_to_document`` failure class):

* ``stage_id`` must resolve to a registered agent-ready stage.
* ``params_of_note`` / ``presets`` / ``constraints`` keys that name a stage
  parameter must actually exist on the constructor.
* ``resource`` uses only known keys with numeric values where numeric is expected.
* a model stage (``model_id`` set) must pin a ``model_version`` (measured-tier).

Measured facts (actual VRAM) are a GPU-CI hook, not checked here; best-guess facts
(``use_cases`` / ``domain``) are shape-checked only. CPU-runnable, no models load.
Run: ``python -m nemo_curator.audio_agent.card_conformance``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

_KNOWN_RESOURCE_KEYS = frozenset(
    {"cpus", "gpu_mem_gb", "host_mem_gb", "gpu_optional", "bound", "throughput_hint", "disk_expansion"}
)
_NUMERIC_RESOURCE_KEYS = frozenset({"cpus", "gpu_mem_gb", "host_mem_gb", "disk_expansion"})
_KNOWN_BOUND = frozenset({"cpu", "gpu", "io"})
_VERIFIED_TIERS = frozenset({"mechanical", "measured", "best_guess"})
_DIRECTIONS = frozenset({"higher_better", "lower_better"})


def _stage_param_names(stage_id: str) -> set[str] | None:
    """Constructor param names of the stage, or None if it doesn't resolve."""
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio._agent_registry import stage_params

    try:
        cls = resolve_stage_class(stage_id)
    except Exception:  # noqa: BLE001 - unknown/unimportable stage
        return None
    try:
        return {p.name for p in stage_params(cls)}
    except Exception:  # noqa: BLE001
        return set()


def _model_version(card: dict[str, Any]) -> Any:  # noqa: ANN401
    return card.get("model_version") or (card.get("provenance") or {}).get("model_version")


_REQUIRED_FIELDS = ("category", "summary", "verified")


def check_card(stage_id: str, card: Any) -> list[str]:  # noqa: ANN401
    """Return a list of mechanical conformance violations for one card (empty = ok)."""
    if not isinstance(card, dict):
        return [f"{stage_id}: card is not a mapping"]
    v: list[str] = []

    params = _stage_param_names(stage_id)
    if params is None:
        return [f"{stage_id}: stage_id does not resolve to a registered agent-ready stage"]

    # required structural fields (a card must state its category, a summary, and how
    # honest each fact is — the verified tiers).
    for f in _REQUIRED_FIELDS:
        if not card.get(f):
            v.append(f"{stage_id}: missing required field {f!r}")

    # params_of_note keys must be real constructor params (the drift catch).
    for k in (card.get("params_of_note") or {}):
        if k not in params:
            v.append(f"{stage_id}: params_of_note lists {k!r} which is not a constructor param of the stage")

    # preset values are param bundles the agent applies as-is: every key must be a real
    # param, else applying the preset would raise bad_params (the asr batch_size drift).
    presets = card.get("presets") or {}
    if isinstance(presets, dict):
        for pname, pvals in presets.items():
            if isinstance(pvals, dict):
                for k in pvals:
                    if k not in params:
                        v.append(f"{stage_id}: preset {pname!r} sets {k!r} which is not a constructor param")

    # resource block: known keys + numeric where expected + valid bound.
    res = card.get("resource") or {}
    if isinstance(res, dict):
        for k, val in res.items():
            if k not in _KNOWN_RESOURCE_KEYS:
                v.append(f"{stage_id}: resource has unknown key {k!r} (allowed: {sorted(_KNOWN_RESOURCE_KEYS)})")
            elif k in _NUMERIC_RESOURCE_KEYS and val is not None and not isinstance(val, (int, float)):
                v.append(f"{stage_id}: resource.{k} must be a number or null, got {val!r}")
            elif k == "bound" and val is not None and val not in _KNOWN_BOUND:
                v.append(f"{stage_id}: resource.bound must be one of {sorted(_KNOWN_BOUND)} or null, got {val!r}")

    # a model stage must pin a model_version (so an upgrade can't silently change facts).
    if card.get("model_id") and not _model_version(card):
        v.append(f"{stage_id}: model_id is set but no model_version pin (add model_version)")

    # metrics block (1A.2): the deterministic source of absolute targets. Validate its
    # shape so the config-strategy resolver can trust it (drift-proof anchors/presets).
    metrics = card.get("metrics") or {}
    if isinstance(metrics, dict):
        for mkey, mblock in metrics.items():
            if not isinstance(mblock, dict):
                v.append(f"{stage_id}: metrics[{mkey!r}] must be a mapping")
                continue
            scale = mblock.get("scale")
            if scale is not None and (not isinstance(scale, dict) or scale.get("direction") not in _DIRECTIONS):
                v.append(f"{stage_id}: metrics[{mkey!r}].scale needs direction in {sorted(_DIRECTIONS)} (+ min/max)")
            tp = mblock.get("threshold_param")
            if tp and tp not in params:
                v.append(f"{stage_id}: metrics[{mkey!r}].threshold_param {tp!r} is not a constructor param")
            for pname, pvals in (mblock.get("presets") or {}).items():
                if isinstance(pvals, dict):
                    for k in pvals:
                        if k not in params:
                            v.append(f"{stage_id}: metrics[{mkey!r}] preset {pname!r} sets non-param {k!r}")
            vr = mblock.get("valid_range")
            if vr is not None and not (isinstance(vr, list) and len(vr) == 2):  # noqa: PLR2004 - [lo, hi]
                v.append(f"{stage_id}: metrics[{mkey!r}].valid_range must be [lo, hi]")

    # verified tiers, when present, must use the known vocabulary.
    verified = card.get("verified") or {}
    if isinstance(verified, dict):
        for fact, tier in verified.items():
            if tier not in _VERIFIED_TIERS:
                v.append(f"{stage_id}: verified[{fact!r}]={tier!r} not in {sorted(_VERIFIED_TIERS)}")

    return v


def audit(index: Any = None) -> dict[str, Any]:  # noqa: ANN401
    """Audit all cards: mechanical violations + coverage (uncarded stages, orphan cards)."""
    from nemo_curator.audio_agent.index import get_index

    idx = index or get_index()
    cards = idx.all_cards()
    violations = {sid: vs for sid, vs in ((sid, check_card(sid, c)) for sid, c in cards.items()) if vs}
    carded = set(cards)
    all_stages = set(idx.stage_names())
    return {
        "violations": violations,
        "orphan_cards": sorted(carded - all_stages),  # cards for a stage that no longer exists
        "uncarded_stages": sorted(all_stages - carded),  # stages still missing a card (coverage gap)
        "carded_count": len(carded),
        "stage_count": len(all_stages),
    }


def main(argv: list[str] | None = None) -> int:
    """Print the audit as JSON; exit non-zero if any card has a mechanical violation."""
    import argparse

    ap = argparse.ArgumentParser(description="Audio-agent capability-card conformance gate")
    ap.add_argument("--allow-uncarded", action="store_true", help="do not fail on coverage gaps (default: report only)")
    args = ap.parse_args(argv)

    result = audit()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    # Drift (violations) or orphan cards are hard failures; coverage gaps are reported only.
    ok = not result["violations"] and not result["orphan_cards"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
