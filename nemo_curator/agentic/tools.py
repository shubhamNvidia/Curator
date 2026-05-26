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
"""The agentic tools — core (NAT-independent) implementations.

These are plain Python callables. The NAT registration shim under
:mod:`nemo_curator.agentic.nat.register` wraps each one as a
``@register_function`` so the agent runtime can invoke them through
tool-calling. Keeping the core logic NAT-free here means the same functions
are directly testable from pytest without bootstrapping the agent runtime.

Tool inventory (the trailing four are used by the team-based planner):

1. ``profile_source_tool``      — Layer 1 dataset characterization.
2. ``capability_search_tool``   — find stages by :class:`CapabilityTag`.
3. ``stage_inspect_tool``       — fetch a stage card by name.
4. ``required_capabilities_tool`` — intent → list of required capability tags.
5. ``gap_report_tool``          — surface intents we cannot satisfy today.
6. ``validate_ir_tool``         — run the 8-check static validator.
7. ``compile_ir_tool``          — emit canonical ``stages:`` YAML.
8. ``run_ir_tool``              — execute (or dry-run) an IR end-to-end.
9. ``dry_run_ir_tool``          — convenience wrapper that forces dry-run.
10. ``deterministic_critic_tool`` — drop-rate + drift summary.
11. ``cache_gc_tool``           — LRU-evict the per-stage cache.
12. ``list_stages_tool``        — list every loaded card (paginated).
13. ``profile_dataset_tool``    — alias of ``profile_source_tool`` returning
    a fully-typed :class:`DatasetCard` (used by team Intent Agent).
14. ``propose_topo_order_tool`` — deterministic phase + data-flow ordering
    of a list of stage names (used by team Staging Agent).
15. ``dataset_quantiles_tool``  — pull p05 / p50 / p95 from a
    :class:`DatasetProfile` so the Tuner can resolve "drop bottom N percent"
    relative thresholds against the actual data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemo_curator.agentic.cards import (
    CapabilityTag,
    DatasetCard,
    RunCard,
    StageCard,
)
from nemo_curator.agentic.compiler import compile_ir_to_yaml
from nemo_curator.agentic.critic import critique
from nemo_curator.agentic.intent import IntentCategories, required_capabilities
from nemo_curator.agentic.ir import PipelineIR
from nemo_curator.agentic.profiler import profile_source
from nemo_curator.agentic.registry import CapabilityRegistry, build_registry
from nemo_curator.agentic.validator import ValidationReport, validate


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def _registry() -> CapabilityRegistry:
    """Cache the registry per-process so tools share one instance."""

    global _CACHED_REGISTRY  # noqa: PLW0603
    if _CACHED_REGISTRY is None:
        _CACHED_REGISTRY = build_registry()
    return _CACHED_REGISTRY


_CACHED_REGISTRY: CapabilityRegistry | None = None


def _coerce_ir(ir_or_path: PipelineIR | dict | str | Path) -> PipelineIR:
    if isinstance(ir_or_path, PipelineIR):
        return ir_or_path
    if isinstance(ir_or_path, dict):
        return PipelineIR.from_dict(ir_or_path)
    if isinstance(ir_or_path, (str, Path)):
        path = Path(ir_or_path)
        if path.exists():
            return PipelineIR.from_path(path)
        return PipelineIR.from_json(str(ir_or_path))
    msg = f"cannot coerce {type(ir_or_path).__name__} into PipelineIR"
    raise TypeError(msg)


def _coerce_intent(intent: IntentCategories | dict | None) -> IntentCategories:
    if isinstance(intent, IntentCategories):
        return intent
    if isinstance(intent, dict):
        return IntentCategories.model_validate(intent)
    return IntentCategories()


# ----------------------------------------------------------------------------
# 1. profile_source_tool
# ----------------------------------------------------------------------------


def profile_source_tool(
    source_uri: str,
    *,
    kind: str = "manifest",
    sample_limit: int = 64,
) -> dict[str, Any]:
    """Profile a source URI and return a JSON-serializable dataset card."""

    from nemo_curator.agentic.adapters import SourceSpec  # noqa: PLC0415

    src = SourceSpec(kind=kind, uri=source_uri)
    card: DatasetCard = profile_source(src, sample_limit=sample_limit)
    return card.model_dump(mode="json")


# ----------------------------------------------------------------------------
# 2. capability_search_tool
# ----------------------------------------------------------------------------


def capability_search_tool(
    capability: str,
    *,
    commercial_only: bool = False,
    include_also_handles: bool = True,
) -> list[dict[str, Any]]:
    """Return every loaded stage that satisfies a capability tag."""

    try:
        tag = CapabilityTag(capability)
    except ValueError as exc:
        msg = (
            f"unknown capability {capability!r}; valid values are: "
            f"{sorted(t.value for t in CapabilityTag)}"
        )
        raise ValueError(msg) from exc

    reg = _registry()
    matches = reg.search_by_capability(
        tag,
        include_also_handles=include_also_handles,
        commercial_only=commercial_only,
    )
    return [_summarize_card(e.card) for e in matches]


# ----------------------------------------------------------------------------
# 3. stage_inspect_tool
# ----------------------------------------------------------------------------


def stage_inspect_tool(name: str) -> dict[str, Any]:
    """Return one full stage card."""

    entry = _registry().get(name)
    if entry is None:
        msg = f"stage {name!r} not in the registry"
        raise KeyError(msg)
    return entry.card.model_dump(mode="json")


# ----------------------------------------------------------------------------
# 4. required_capabilities_tool
# ----------------------------------------------------------------------------


def required_capabilities_tool(intent: IntentCategories | dict | None) -> list[dict[str, Any]]:
    """Return the capability requirements implied by an intent."""

    ic = _coerce_intent(intent)
    return [
        {
            "intent_field": req.intent_field,
            "capability": req.capability.value,
            "rationale": req.reason,
            "required": req.required,
        }
        for req in required_capabilities(ic)
    ]


# ----------------------------------------------------------------------------
# 5. gap_report_tool
# ----------------------------------------------------------------------------


def gap_report_tool(intent: IntentCategories | dict | None) -> dict[str, Any]:
    """Return the subset of required capabilities the catalog cannot satisfy.

    The agent uses this as the source of truth for refusing or partially
    honoring a prompt. Each gap carries the schema for *future* stages so
    the user knows what they're missing today.
    """

    ic = _coerce_intent(intent)
    reqs = required_capabilities(ic)
    reg = _registry()

    supported: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []

    for req in reqs:
        cands = reg.search_by_capability(req.capability)
        if cands:
            supported.append({
                "capability": req.capability.value,
                "candidates": [c.card.name for c in cands],
            })
        else:
            gaps.append({
                "capability": req.capability.value,
                "intent_field": req.intent_field,
                "rationale": req.reason,
                "fix_hint": "This capability is not in the current catalog. "
                            "Phase 5 will add it via the wizard onboarding flow.",
            })

    return {"supported": supported, "gaps": gaps, "has_gaps": bool(gaps)}


# ----------------------------------------------------------------------------
# 6. validate_ir_tool
# ----------------------------------------------------------------------------


def validate_ir_tool(
    ir: PipelineIR | dict | str | Path,
    *,
    mutate: bool = True,
) -> dict[str, Any]:
    """Run the 8-check static validator."""

    pipeline_ir = _coerce_ir(ir)
    report: ValidationReport = validate(pipeline_ir, _registry(), mutate=mutate)
    return {
        "ok": report.is_ok(),
        "fingerprint": report.fingerprint,
        "findings": [
            {
                "severity": f.severity.value,
                "code": f.code,
                "detail": f.detail,
                "stage_index": f.stage_index,
                "stage_name": f.stage_name,
            }
            for f in report.findings
        ],
        "auto_inserted": [s.model_dump(mode="json") for s in report.auto_inserted],
        "ir": report.ir.to_dict(),
    }


# ----------------------------------------------------------------------------
# 7. compile_ir_tool
# ----------------------------------------------------------------------------


def compile_ir_tool(ir: PipelineIR | dict | str | Path) -> str:
    """Compile an IR to canonical Hydra-style YAML."""

    return compile_ir_to_yaml(_coerce_ir(ir), _registry())


# ----------------------------------------------------------------------------
# 8 + 9. run_ir_tool / dry_run_ir_tool
# ----------------------------------------------------------------------------


def run_ir_tool(
    ir: PipelineIR | dict | str | Path,
    *,
    target_dir: str | None = None,
    dry_run: bool = False,
    enable_cache: bool = True,
) -> dict[str, Any]:
    """Run (or dry-run) an IR; returns the resulting RunCard as JSON."""

    from nemo_curator.agentic.runner import RunOptions, run  # noqa: PLC0415

    pipeline_ir = _coerce_ir(ir)
    if target_dir:
        pipeline_ir = pipeline_ir.model_copy(
            update={"sink": pipeline_ir.sink.model_copy(update={"target_dir": target_dir})},
        )
    opts = RunOptions(dry_run=dry_run, enable_cache=enable_cache)
    result = run(pipeline_ir, _registry(), options=opts)
    return {
        "success": result.success,
        "run_card": result.run_card.model_dump(mode="json"),
        "target_dir": str(result.target_dir),
        "compiled_yaml_path": str(result.compiled_yaml_path),
        "error": result.error,
    }


def dry_run_ir_tool(ir: PipelineIR | dict | str | Path) -> dict[str, Any]:
    """Convenience wrapper that always forces dry-run."""

    return run_ir_tool(ir, dry_run=True)


# ----------------------------------------------------------------------------
# 10. deterministic_critic_tool
# ----------------------------------------------------------------------------


def deterministic_critic_tool(
    run_card: RunCard | dict | str | Path,
    *,
    intent: IntentCategories | dict | None = None,
    input_card: DatasetCard | dict | None = None,
    output_card: DatasetCard | dict | None = None,
) -> dict[str, Any]:
    """Run the LLM-free critic on a completed RunCard."""

    rc = _coerce_run_card(run_card)
    in_card = _coerce_dataset_card(input_card)
    out_card = _coerce_dataset_card(output_card)
    rep = critique(
        run_card=rc,
        intent=_coerce_intent(intent) if intent else None,
        input_card=in_card,
        output_card=out_card,
    )
    return rep.model_dump(mode="json")


# ----------------------------------------------------------------------------
# 11. cache_gc_tool
# ----------------------------------------------------------------------------


def cache_gc_tool(target_dir: str, *, max_bytes: int = 50 * 1024 * 1024 * 1024) -> dict[str, Any]:
    """LRU-evict the per-stage cache under ``<target_dir>/.adv/cache``."""

    from nemo_curator.agentic.cache import StageCache  # noqa: PLC0415

    cache_dir = Path(target_dir) / ".adv" / "cache"
    cache = StageCache(cache_dir, max_bytes=max_bytes)
    before = cache.total_bytes()
    freed = cache.gc(target_bytes=max_bytes)
    after = cache.total_bytes()
    return {"bytes_before": before, "bytes_freed": freed, "bytes_after": after}


# ----------------------------------------------------------------------------
# 12. list_stages_tool
# ----------------------------------------------------------------------------


def list_stages_tool(
    *,
    category: str | None = None,
    commercial_only: bool = False,
    limit: int = 64,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated stage listing (used when the agent wants the catalog overview)."""

    reg = _registry()
    entries = list(reg.by_name.values())
    if category:
        entries = [e for e in entries if e.card.category.value == category]
    if commercial_only:
        entries = [e for e in entries if e.card.commercial_safe]
    entries.sort(key=lambda e: (e.card.category.value, e.card.name))
    total = len(entries)
    window = entries[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "stages": [_summarize_card(e.card) for e in window],
    }


# ----------------------------------------------------------------------------
# 13. profile_dataset_tool  (team Intent Agent)
# ----------------------------------------------------------------------------


def profile_dataset_tool(
    source_uri: str,
    *,
    kind: str = "manifest",
    sample_limit: int = 64,
) -> DatasetCard:
    """Profile a source URI and return a fully-typed :class:`DatasetCard`.

    Same underlying call as :func:`profile_source_tool` but returns the
    Pydantic model (not a dict) so the Python-side team manager can
    pass it around without re-validating.
    """

    from nemo_curator.agentic.adapters import SourceSpec  # noqa: PLC0415

    src = SourceSpec(kind=kind, uri=source_uri)
    return profile_source(src, sample_limit=sample_limit)


# ----------------------------------------------------------------------------
# 14. propose_topo_order_tool  (team Staging Agent)
# ----------------------------------------------------------------------------


def propose_topo_order_tool(stage_names: list[str]) -> dict[str, Any]:
    """Return a deterministic phase-and-data-flow ordering of ``stage_names``.

    Algorithm:

    1. Resolve each stage to its card. Unknown names are dropped from the
       output and reported in ``unknown``.
    2. Group by :class:`Phase` and sort phases by their canonical rank.
    3. Within each phase, use the user-provided order (stable).
    4. Re-walk the resulting sequence and bubble each consumer down past
       any later producer of a key it needs (one pass; sufficient because
       phase ordering already covers the gross structure).

    The team Staging Agent calls this BEFORE proposing its final order so
    it gets a strong deterministic suggestion to start from; it can still
    override but typically does not need to.
    """

    from nemo_curator.agentic.cards import PHASE_ORDER, Phase  # noqa: PLC0415

    reg = _registry()
    known: list[tuple[str, Any]] = []
    unknown: list[str] = []
    for name in stage_names:
        entry = reg.get(name)
        if entry is None:
            unknown.append(name)
        else:
            known.append((name, entry.card))

    # Phase grouping (stable within-phase).
    def _phase_rank(card: Any) -> int:
        return PHASE_ORDER.get(card.phase, PHASE_ORDER[Phase.ANALYZE])

    indexed = list(enumerate(known))
    indexed.sort(key=lambda pair: (_phase_rank(pair[1][1]), pair[0]))

    ordered_names = [name for _, (name, _card) in indexed]
    ordered_cards = [card for _, (_name, card) in indexed]

    # One forward pass producer-before-consumer correction within phases.
    produces: list[set[str]] = [
        set(c.outputs.data) | set(c.produces_keys_after_run) for c in ordered_cards
    ]
    requires: list[set[str]] = [set(c.inputs.data) for c in ordered_cards]
    available: set[str] = {"audio_filepath", "task_id"}
    fixed = list(ordered_names)
    for i, _ in enumerate(fixed):
        needed = requires[i] - available
        if needed:
            for j in range(i + 1, len(fixed)):
                if produces[j] & needed:
                    # Move j before i.
                    fixed.insert(i, fixed.pop(j))
                    produces.insert(i, produces.pop(j))
                    requires.insert(i, requires.pop(j))
                    available |= produces[i]
                    break
            else:
                available |= produces[i]
        else:
            available |= produces[i]

    return {
        "ordered_stages": fixed,
        "unknown": unknown,
        "by_phase": {
            phase.value: [n for n, c in zip(fixed, ordered_cards) if c.phase == phase]
            for phase in Phase
        },
    }


# ----------------------------------------------------------------------------
# 15. dataset_quantiles_tool  (team Param Tuner)
# ----------------------------------------------------------------------------


def dataset_quantiles_tool(
    profile: DatasetCard | dict | None,
    *,
    metric: str = "duration",
) -> dict[str, float | None]:
    """Return ``{p05, p50, p95}`` for a metric on the dataset profile.

    Currently the profiler captures only the duration distribution
    (:attr:`DatasetProfile.duration_p05_sec` etc.). Other metrics — MOS,
    SNR, bandwidth — will be added when their per-clip scorers ship; the
    tool already returns ``None`` rather than failing so the Tuner can
    degrade gracefully.
    """

    dc = _coerce_dataset_card(profile)
    if dc is None:
        return {"p05": None, "p50": None, "p95": None, "metric": metric}
    if metric == "duration":
        return {
            "p05": dc.profile.duration_p05_sec,
            "p50": dc.profile.duration_p50_sec,
            "p95": dc.profile.duration_p95_sec,
            "metric": "duration",
        }
    return {"p05": None, "p50": None, "p95": None, "metric": metric}


# ----------------------------------------------------------------------------
# Card summarization (kept lightweight to fit in LLM context)
# ----------------------------------------------------------------------------


def _summarize_card(card: StageCard) -> dict[str, Any]:
    return {
        "name": card.name,
        "category": card.category.value,
        "summary": card.summary,
        "capabilities": [t.value for t in card.capabilities],
        "also_handles": [t.value for t in card.also_handles],
        "produces_cardinality": card.produces_cardinality.value,
        "cost_hint": card.cost_hint,
        "commercial_safe": card.commercial_safe,
        "license": card.license.value,
        "requires_sample_rate": card.requires_sample_rate,
        "requires_in_memory_waveform": card.requires_in_memory_waveform,
    }


def _coerce_run_card(rc: RunCard | dict | str | Path) -> RunCard:
    if isinstance(rc, RunCard):
        return rc
    if isinstance(rc, dict):
        return RunCard.model_validate(rc)
    path = Path(rc)
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # noqa: PLC0415

        return RunCard.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return RunCard.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _coerce_dataset_card(dc: DatasetCard | dict | None) -> DatasetCard | None:
    if dc is None:
        return None
    if isinstance(dc, DatasetCard):
        return dc
    return DatasetCard.model_validate(dc)


# ----------------------------------------------------------------------------
# Public registry (used by the NAT shim and the test suite)
# ----------------------------------------------------------------------------


TOOLS: dict[str, Any] = {
    "profile_source": profile_source_tool,
    "capability_search": capability_search_tool,
    "stage_inspect": stage_inspect_tool,
    "required_capabilities": required_capabilities_tool,
    "gap_report": gap_report_tool,
    "validate_ir": validate_ir_tool,
    "compile_ir": compile_ir_tool,
    "run_ir": run_ir_tool,
    "dry_run_ir": dry_run_ir_tool,
    "deterministic_critic": deterministic_critic_tool,
    "cache_gc": cache_gc_tool,
    "list_stages": list_stages_tool,
    "profile_dataset": profile_dataset_tool,
    "propose_topo_order": propose_topo_order_tool,
    "dataset_quantiles": dataset_quantiles_tool,
}


__all__ = [
    "TOOLS",
    "cache_gc_tool",
    "capability_search_tool",
    "compile_ir_tool",
    "dataset_quantiles_tool",
    "deterministic_critic_tool",
    "dry_run_ir_tool",
    "gap_report_tool",
    "list_stages_tool",
    "profile_dataset_tool",
    "profile_source_tool",
    "propose_topo_order_tool",
    "required_capabilities_tool",
    "run_ir_tool",
    "stage_inspect_tool",
    "validate_ir_tool",
]
