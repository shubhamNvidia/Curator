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
* ``params_of_note`` / ``presets`` keys that name a stage parameter must actually
  exist on the constructor. (``constraints`` are model facts, e.g.
  ``supported_sample_rates`` / ``max_speakers`` -- NOT constructor params -- so they
  are intentionally not checked against the signature.)
* ``resource`` uses only known keys with numeric values where numeric is expected.
* a model stage (``model_id`` set) must pin a ``model_version`` (measured-tier).
* a ``metrics`` block, if present, must use a valid ``scale.direction``, a real
  ``threshold_param``, and a ``[lo, hi]`` ``valid_range``.
* ``semantic_facts``, if present, is shape-checked as advisory prose.  The gate
  never interprets scope or turns a fact into a module-specific pipeline rule.
* a capability ``tag`` must reflect the stage's DEFAULT behavior: a tag that maps to an
  unconditional boolean contract gate (``writes_disk``/``needs_ffmpeg``) is checked against
  a default-constructed instance's gate. An opt-in capability belongs in a param (knob),
  not a tag. (``needs_gpu`` and ``needs_internet_first_run`` are intentionally NOT checked:
  their gates are conditional -- ``resources.gpus > 0`` and ``model_path/auto_download`` --
  so they aren't a clean tag<->gate equality.)

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
_SEMANTIC_PROSE_FIELDS = frozenset({"meaning", "unit", "provenance", "scope", "propagation"})


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


def _semantic_fact_violations(stage_id: str, raw: Any) -> list[str]:  # noqa: ANN401
    """Validate only the JSON/YAML shape of optional semantic reasoning prose.

    A compact string and a richer mapping are both accepted.  Meaning, scope,
    and propagation remain free text: conformance can ensure the packet is
    readable, but only a reviewer can judge whether it matches user intent.
    """
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{stage_id}: semantic_facts must be a mapping"]
    violations: list[str] = []
    for anchor, fact in raw.items():
        if not isinstance(anchor, str) or not anchor.strip():
            violations.append(f"{stage_id}: semantic_facts keys must be non-empty strings")
            continue
        prefix = f"{stage_id}: semantic_facts[{anchor!r}]"
        if isinstance(fact, str):
            if not fact.strip():
                violations.append(f"{prefix} must not be empty")
            continue
        if not isinstance(fact, dict):
            violations.append(f"{prefix} must be prose or a mapping")
            continue
        for field in _SEMANTIC_PROSE_FIELDS:
            if field in fact and (not isinstance(fact[field], str) or not fact[field].strip()):
                violations.append(f"{prefix}.{field} must be a non-empty string")
        counterexamples = fact.get("counterexamples")
        if counterexamples is not None and (
            not isinstance(counterexamples, list)
            or not counterexamples
            or any(not isinstance(item, str) or not item.strip() for item in counterexamples)
        ):
            violations.append(f"{prefix}.counterexamples must be a non-empty list of non-empty strings")
    return violations


# Capability tag -> the boolean contract gate it must mirror. A tag states DEFAULT behavior,
# so we check it against a DEFAULT-constructed instance's gate. Only tags whose gate is an
# UNCONDITIONAL boolean are included:
#   * ``needs_gpu`` is excluded -- its gate is ``resources.gpus > 0`` (true for gpu_optional
#     stages that rightly omit the tag), not a clean tag<->gate equality.
#   * ``needs_internet_first_run`` is excluded -- its gate is often conditional on a knob
#     (``model_path is None`` / ``auto_download``), so "default" is ambiguous.
# Other tags (produces_score, is_filter, fanout, sink, batch_only, needs_hf_token) are
# structural/role facts, not boolean gates.
_TAG_GATES: dict[str, str] = {
    "writes_disk": "writes_to_disk",
    "needs_ffmpeg": "requires_ffmpeg",
}


def _effective_default_gates(stage_id: str) -> dict[str, bool] | None:
    """DEFAULT-constructed gate values for the tag-checked attrs, or ``None`` if the stage
    can't be built cheaply (required args / build error).

    A composite hides its own gates (``wrappable=False``), so its *effective* gate is the OR
    of its decomposed inner stages' gates -- e.g. SplitASRAlignJoin writes to disk because its
    inner SplitLongAudioStage does, even though the composite's own contract declares nothing.
    Best-effort: never raises.
    """
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.base import CompositeStage

    attrs = set(_TAG_GATES.values())
    try:
        inst = resolve_stage_class(stage_id)()
        contract = foundation.build_contract(inst)
        if not contract.wrappable and isinstance(inst, CompositeStage):
            inner = [foundation.build_contract(s).gates for s in inst.decompose()]
            return {a: any(bool(getattr(g, a, False)) for g in inner) for a in attrs}
        return {a: bool(getattr(contract.gates, a, False)) for a in attrs}
    except Exception:  # noqa: BLE001 - not default-buildable -> can't verify (skip, not a failure)
        return None


def _tag_gate_violations(stage_id: str, card: dict[str, Any]) -> list[str]:
    """Tags that claim a capability the stage does NOT do by default.

    Forward-only (tag present -> default gate must be True): a stage may legitimately have a
    gate on by default without the tag (e.g. gpu_optional), so we do not flag the reverse.
    Composites are judged by the OR of their inner stages' gates; stages that aren't
    default-buildable are skipped (can't verify), never failed.
    """
    checkable = {t for t in (card.get("tags") or []) if t in _TAG_GATES}
    if not checkable:
        return []
    gates = _effective_default_gates(stage_id)
    if gates is None:
        return []
    out: list[str] = []
    for tag in sorted(checkable):
        attr = _TAG_GATES[tag]
        if not gates.get(attr, False):
            out.append(
                f"{stage_id}: card tag {tag!r} but the stage's DEFAULT contract gate {attr}=False "
                f"-- a tag states DEFAULT behavior; make this an opt-in param (knob) instead of a tag, "
                f"or fix the gate"
            )
    return out


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

    # Semantic facts are retrieval material for the host critic.  Validate
    # shape only; do not encode field meaning or scope into deterministic rules.
    v.extend(_semantic_fact_violations(stage_id, card.get("semantic_facts")))

    # versions block (optional, model-backed stages): {model_id: "when-to-use"} for
    # checkpoints verified interchangeable via model_name/model_path (same output structure,
    # no module code change). Keep it honest + drift-proof: a version an agent can *select*
    # (a preset that sets model_name/model_path) must be documented here.
    versions = card.get("versions")
    if versions is not None:
        if not isinstance(versions, dict) or not all(
            isinstance(mid, str) and isinstance(desc, str) for mid, desc in versions.items()
        ):
            v.append(f"{stage_id}: versions must be a mapping of {{model_id: 'when-to-use string'}}")
        else:
            if not card.get("model_id"):
                v.append(f"{stage_id}: versions is set but model_id is null (versions document model checkpoints)")
            if isinstance(presets, dict):
                for pname, pvals in presets.items():
                    if not isinstance(pvals, dict):
                        continue
                    for mk in ("model_name", "model_path"):
                        if mk in pvals and pvals[mk] not in versions:
                            v.append(
                                f"{stage_id}: preset {pname!r} selects {mk}={pvals[mk]!r} which is not documented in versions"
                            )

    # verified tiers, when present, must use the known vocabulary.
    verified = card.get("verified")
    if not isinstance(verified, dict):
        v.append(f"{stage_id}: verified must be a mapping of fact names to evidence tiers")
    else:
        for fact, tier in verified.items():
            if tier not in _VERIFIED_TIERS:
                v.append(f"{stage_id}: verified[{fact!r}]={tier!r} not in {sorted(_VERIFIED_TIERS)}")
        if card.get("semantic_facts") is not None and "semantic_facts" not in verified:
            v.append(
                f"{stage_id}: semantic_facts must declare its evidence tier in "
                "verified.semantic_facts"
            )

    # tag <-> default-gate consistency (M5b): a capability tag must reflect DEFAULT behavior.
    v.extend(_tag_gate_violations(stage_id, card))

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
