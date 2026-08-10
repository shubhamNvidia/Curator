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


def _filter_tag_violations(stage_id: str, card: dict[str, Any]) -> list[str]:
    """A contract declaring ``cardinality="filter"`` must carry the ``is_filter`` tag.

    The contract is the stricter statement and the one a stage author is most likely to write
    alone, having just made the stage drop rows. Without the tag nothing assembling a recipe
    knows it can: a stage that silently discards most of a corpus reads as a pass-through
    exactly where the decision to include it is made.

    Only this direction. The tag is the broader planner-facing notion -- ``OverlapFilterStage``
    and ``ALMDataOverlapStage`` filter WITHIN a row, shrinking a segment list while every row
    survives -- so a tag without ``cardinality="filter"`` is a correct pairing, not a drift.
    Checking the converse would demand those stages declare a row cardinality they do not have.
    """
    if "is_filter" in (card.get("tags") or []):
        return []
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio import agent as foundation

    try:
        contract = foundation.build_contract(resolve_stage_class(stage_id)())
    except Exception:  # noqa: BLE001 - not default-buildable -> can't verify (skip, not a failure)
        return []
    if getattr(contract, "cardinality", None) != "filter":
        return []
    return [
        f"{stage_id}: the stage's DEFAULT contract declares cardinality='filter' but the card "
        f"has no 'is_filter' tag, so nothing planning a recipe knows this stage can drop rows"
    ]


_REQUIRED_FIELDS = ("category", "summary", "verified")

# Every top-level key a card may carry. A closed vocabulary because the failure it prevents is
# silent: a card key nobody reads is not an error anywhere, it is simply absent from the packet
# the host critic sees, and the card still passes conformance and still says ``validated``. Two
# shipped cards wrote ``gotchas`` and ``relationships`` for what the readers call
# ``counterexamples`` and ``comparison``, so carefully written disambiguation prose -- the exact
# material meant to stop a stage being confused with its neighbour -- reached nobody at all.
# Adding a key here is the deliberate half of adding a reader for it.
_KNOWN_CARD_FIELDS = frozenset(
    {
        "stage_id", "category", "summary", "tags", "model_id", "model_version", "domain",
        "constraints", "resource", "use_cases", "composition", "verified", "params_of_note",
        "provenance", "notes", "param_dependencies", "comparison", "semantic_facts",
        "conflicts_with", "presets", "caveats", "metrics", "versions", "deterministic",
    }
)


def _composition_violations(stage_id: str, card: dict[str, Any]) -> list[str]:
    """Check the stages a card recommends chaining with actually exist and don't contradict.

    ``composition`` is the part of a card an agent acts on most directly -- it is read as "these
    are the stages to put either side of this one" -- so a wrong name here is worse than a
    missing one. ``ALMDataBuilderStage`` recommended ``PrepareModuleSegmentsStage`` upstream for
    two card versions while that pairing raised ``TypeError`` on the first window, because the
    one writes ``metrics.bandwidth`` as a per-word list and the other compares it to an int.
    Nothing caught it: the recommendation was prose pointing at a name.

    Full edge simulation was considered and rejected -- building two default-constructed stages
    and validating them flags every pair needing params or a seed, and a gate that cries wolf
    gets ignored. These are the checks that cannot false-positive: a name must resolve, and a
    stage cannot be recommended and forbidden at once.
    """
    comp = card.get("composition")
    if not isinstance(comp, dict):
        return []
    v: list[str] = []
    typical: set[str] = set()
    for field in ("typical_upstream", "typical_downstream"):
        names = comp.get(field) or []
        if not isinstance(names, list):
            v.append(f"{stage_id}: composition.{field} must be a list of stage ids")
            continue
        if field == "typical_upstream":
            typical |= set(names)
        v += [
            f"{stage_id}: composition.{field} names {name!r}, which is not a registered stage"
            for name in names
            if _stage_param_names(str(name)) is None
        ]
    return v + _incompatible_violations(stage_id, comp, typical)


def _incompatible_violations(stage_id: str, comp: dict[str, Any], typical: set[str]) -> list[str]:
    """Violations in the optional ``incompatible_upstream`` map."""
    incompatible = comp.get("incompatible_upstream") or {}
    if not isinstance(incompatible, dict):
        return [f"{stage_id}: composition.incompatible_upstream must be a mapping of stage id -> reason"]
    v: list[str] = []
    for name, reason in incompatible.items():
        if _stage_param_names(str(name)) is None:
            v.append(f"{stage_id}: composition.incompatible_upstream names {name!r}, which is not a registered stage")
        if not str(reason or "").strip():
            v.append(f"{stage_id}: composition.incompatible_upstream[{name!r}] must say WHY, not just name the stage")
        if name in typical:
            v.append(f"{stage_id}: composition lists {name!r} as both typical_upstream and incompatible_upstream")
    return v


def _composite_legibility(stage_id: str) -> list[str]:
    """A composite must reveal its stages, or declare its own I/O; silence is not an option.

    A ``CompositeStage`` whose ``describe()`` returns a bare ``StageContract(wrappable=False)``
    tells a reader it has no reads and no writes, which is not the same as having unknown ones.
    Validation opens composites now, so this holds the door open: a new composite that neither
    decomposes at plan time nor states its own contract reintroduces the blind spot that let a
    ``segments``/``diar_segments`` mismatch survive validation and fail after two model
    downloads and a GPU diarization pass.
    """
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio._composite import expand_composites

    try:
        cls = resolve_stage_class(stage_id)
        from nemo_curator.stages.base import CompositeStage

        if not (isinstance(cls, type) and issubclass(cls, CompositeStage)):
            return []
        instance = cls()
    except Exception:  # noqa: BLE001 - a composite needing constructor args is exercised by its own tests
        return []
    if expand_composites([instance]).fully_resolved:
        return []
    contract = getattr(instance, "describe", lambda: None)()
    if contract is not None and (contract.reads.data_keys or contract.writes.data_keys):
        return []
    return [
        (
            f"{stage_id}: composite neither decomposes at plan time nor declares its own "
            f"reads/writes, so nothing downstream of it can be validated"
        ),
    ]


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

    v += [
        f"{stage_id}: unknown top-level field {f!r}; nothing reads it, so its content reaches "
        f"nobody (allowed: {sorted(_KNOWN_CARD_FIELDS)})"
        for f in sorted(card)
        if f not in _KNOWN_CARD_FIELDS
    ]

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

    v.extend(_composition_violations(stage_id, card))
    v.extend(_composite_legibility(stage_id))

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
    v.extend(_filter_tag_violations(stage_id, card))

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


def _blueprint_violations(blueprint_id: str, blueprint: Any) -> list[str]:  # noqa: ANN401
    """Check a blueprint's stage refs resolve and its presets name real parameters.

    Blueprints are shown to the planner as worked examples, and its presets are read as
    "these are the knobs for this pipeline" -- so a preset naming a parameter that does not
    exist is a confident instruction to do something impossible. Every shipped preset was
    wrong this way (``utmos_mos_threshold`` for what the stage calls ``mos_threshold``,
    ``wer_threshold`` for ``target_value``), and nothing caught it: the conformance gate
    covered cards only, while the blueprints declared ``validated: true``.

    A preset parameter is accepted when SOME stage the blueprint lists accepts it -- the
    presets are pipeline-level, so they are not attributable to one stage.
    """
    if not isinstance(blueprint, dict):
        return [f"{blueprint_id}: blueprint is not a mapping"]
    v: list[str] = []
    refs = [str(s.get("ref")) for s in (blueprint.get("stages") or []) if isinstance(s, dict) and s.get("ref")]
    accepted: set[str] = set()
    for ref in refs:
        params = _stage_param_names(ref)
        if params is None:
            v.append(f"{blueprint_id}: stages names {ref!r}, which is not a registered stage")
            continue
        accepted |= params
    presets = blueprint.get("presets") or {}
    if not isinstance(presets, dict):
        return [*v, f"{blueprint_id}: presets must be a mapping of name -> {{param: value}}"]
    for name, values in presets.items():
        if not isinstance(values, dict):
            v.append(f"{blueprint_id}: preset {name!r} must be a mapping of {{param: value}}")
            continue
        v += [
            f"{blueprint_id}: preset {name!r} sets {param!r}, which no stage in this blueprint accepts"
            for param in values
            if refs and param not in accepted
        ]
    return v


def audit_blueprints(index: Any = None) -> dict[str, Any]:  # noqa: ANN401
    """Mechanical violations across every blueprint (same contract as :func:`audit`)."""
    from nemo_curator.audio_agent.index import get_index

    idx = index or get_index()
    violations: dict[str, list[str]] = {}
    for blueprint in idx.blueprints():
        blueprint_id = str((blueprint or {}).get("blueprint_id") or "<unnamed>")
        found = _blueprint_violations(blueprint_id, blueprint)
        if found:
            violations[blueprint_id] = found
    return {"violations": violations, "blueprint_count": len(idx.blueprints())}


def main(argv: list[str] | None = None) -> int:
    """Print the audit as JSON; exit non-zero if any card has a mechanical violation."""
    import argparse

    ap = argparse.ArgumentParser(description="Audio-agent capability-card conformance gate")
    ap.add_argument("--allow-uncarded", action="store_true", help="do not fail on coverage gaps (default: report only)")
    args = ap.parse_args(argv)

    result = audit()
    # Blueprints are planner-facing worked examples, so a preset naming a parameter that
    # does not exist misleads exactly like a drifted card; gate them the same way.
    blueprints = audit_blueprints()
    result["blueprint_violations"] = blueprints["violations"]
    result["blueprint_count"] = blueprints["blueprint_count"]
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    # Drift (violations) or orphan cards are hard failures; coverage gaps are reported only.
    ok = not result["violations"] and not result["orphan_cards"] and not blueprints["violations"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
