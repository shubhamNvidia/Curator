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
"""Multi-agent Planner DAG.

Replaces the single React-agent loop with a small DAG of focused LLM calls
glued together by deterministic Python. Each LLM step has exactly one job;
the validator / topo-sort / compiler stay where they were.

Pipeline (read top-to-bottom):

    user prompt + dataset profile
        │
        ▼
    Step 1: extract_intent(prompt, profile, llm) -> IntentCategories      (LLM)
        │
        ▼
    Step 2: required_capabilities(intent) -> list[CapabilityRequirement]  (deterministic)
        │
        ▼
    Step 3: pick_stages(caps, registry, llm)    -> list[str stage names]  (LLM, one call per cap)
        │
        ▼
    Step 4: tune_params(stage_names, intent, registry, llm) -> list[StageRef]  (LLM, one call per stage)
        │
        ▼
    Step 5: validator.validate(ir, registry, mutate=True)                 (deterministic)
        │
        ▼
    Step 6: critic(ir, prompt, intent, profile, llm) -> approve|refine    (LLM, bounded loop)
        │
        ▼
    compile_ir_to_yaml(ir, registry)                                       (deterministic)


This module deliberately keeps the LLM-facing surface tiny: each helper
takes a :class:`LLMClient` and a small dict-shaped context, returns a typed
Pydantic object. Mock LLMs slot in for tests without touching the network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_curator.agentic.adapters import probe_source
from nemo_curator.agentic.cards import CapabilityTag, DatasetCard, StageCard
from nemo_curator.agentic.intent import (
    CapabilityRequirement,
    IntentCategories,
    required_capabilities,
)
from nemo_curator.agentic.ir import PipelineIR, SinkSpec, SourceSpec, StageRef
from nemo_curator.agentic.llm import LLMClient, Message
from nemo_curator.agentic.profiler import profile_source
from nemo_curator.agentic.registry import CapabilityRegistry
from nemo_curator.agentic.validator import ValidationReport, validate

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------


@dataclass
class CriticVerdict:
    """Result of one critic turn — approve or refine."""

    approved: bool
    complaints: list[str] = field(default_factory=list)
    patch: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass
class PlannerResult:
    """Bundle returned from :func:`plan`."""

    ir: PipelineIR
    report: ValidationReport
    intent: IntentCategories
    profile: DatasetCard | None
    critic_history: list[CriticVerdict]


# ----------------------------------------------------------------------------
# Prompt loader
# ----------------------------------------------------------------------------


_PROMPTS_DIR = Path(__file__).parent / "nat" / "prompts"


def _load_prompt(name: str) -> str:
    """Read one of the role prompts from ``nat/prompts/<name>.md``.

    Cached at first use; missing prompt = ValueError (not silently
    returning empty so we'd hide a real misconfiguration).
    """

    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        msg = (
            f"planner DAG prompt {name!r} not found at {path}. The prompts "
            f"directory is created in the multi-agent DAG plan; if you "
            f"see this, the install is incomplete."
        )
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------
# Step 1: Intent Extractor (LLM)
# ----------------------------------------------------------------------------


def extract_intent(
    prompt: str,
    profile: DatasetCard | None,
    llm: LLMClient,
    *,
    tier: str = "synth",
) -> IntentCategories:
    """Translate a free-form prompt + dataset profile into :class:`IntentCategories`.

    Single-purpose LLM call. The system prompt explicitly maps common
    adjectives ("clean", "studio", "noisy") to the schema's numeric fields
    so the agent can't accidentally drop them. On malformed JSON we retry
    once at ``temperature=0``; second failure raises ``ValueError``.
    """

    sys_prompt = _load_prompt("extractor")
    profile_dict = profile.model_dump(mode="json") if profile else None
    payload = {
        "user_prompt": prompt,
        "dataset_profile": profile_dict,
        "schema_fields_summary": _intent_schema_summary(),
    }
    messages = [
        Message("system", sys_prompt),
        Message("user", json.dumps(payload, indent=2)),
    ]
    raw = _call_json(llm, messages, tier=tier, purpose="extractor")
    try:
        return IntentCategories.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        # The schema uses extra="ignore" so most stray fields are tolerated;
        # only an unrecoverable type error reaches here.
        msg = f"Step 1 (extract_intent) returned invalid IntentCategories: {exc}; raw={raw!r}"
        raise ValueError(msg) from exc


def _intent_schema_summary() -> dict[str, Any]:
    """A compact field-by-field schema description handed to the LLM.

    Kept here (not in the prompt) so it tracks :class:`IntentCategories`
    automatically as the schema evolves.
    """

    sch = IntentCategories.model_json_schema()
    fields: dict[str, str] = {}
    for fname, fmeta in sch.get("properties", {}).items():
        desc = fmeta.get("description") or ""
        fields[fname] = desc
    return fields


# ----------------------------------------------------------------------------
# Step 2: required_capabilities (deterministic — wrapped for symmetry)
# ----------------------------------------------------------------------------


def derive_capabilities(intent: IntentCategories) -> list[CapabilityRequirement]:
    """Deterministic intent → capabilities mapping. Thin wrapper for symmetry."""

    return required_capabilities(intent)


# ----------------------------------------------------------------------------
# Step 3: Stage Picker (LLM, one call per capability)
# ----------------------------------------------------------------------------


def pick_stages(
    caps: list[CapabilityRequirement],
    registry: CapabilityRegistry,
    llm: LLMClient,
    *,
    tier: str = "synth",
) -> list[str]:
    """For each required capability, pick one or more complementary stages.

    Most capabilities collapse to a single stage. Complementary families
    (e.g. ``quality_filter_mos`` → UTMOS + SIGMOS) may return multiple.
    Capabilities for which the registry has no candidate are skipped
    (gap_report would have refused upstream).
    De-duplicates across capabilities while preserving order.
    """

    sys_prompt = _load_prompt("picker")
    picked: list[str] = []
    seen: set[str] = set()
    for req in caps:
        candidates = registry.search_by_capability(req.capability)
        if not candidates:
            logger.warning(
                "Step 3 (pick_stages): no candidates for capability %s — skipping",
                req.capability.value,
            )
            continue
        if len(candidates) == 1:
            chosen = [candidates[0].card.name]
        else:
            chosen = _llm_pick(req, candidates, llm, sys_prompt, tier=tier)
        for name in chosen:
            if name not in seen:
                picked.append(name)
                seen.add(name)
    return picked


def _llm_pick(
    req: CapabilityRequirement,
    candidates: list[Any],
    llm: LLMClient,
    sys_prompt: str,
    *,
    tier: str,
) -> list[str]:
    candidate_summary = [_card_summary_for_picker(e.card) for e in candidates]
    payload = {
        "capability": req.capability.value,
        "reason": req.reason,
        "candidates": candidate_summary,
    }
    messages = [
        Message("system", sys_prompt),
        Message("user", json.dumps(payload, indent=2)),
    ]
    raw = _call_json(llm, messages, tier=tier)
    # Accept ``chosen_stages`` (list, preferred) or ``chosen_stage`` (str,
    # legacy from earlier versions of the picker prompt).
    chosen = raw.get("chosen_stages")
    if not chosen and "chosen_stage" in raw:
        chosen = [raw["chosen_stage"]]
    if not isinstance(chosen, list) or not chosen:
        logger.warning(
            "Step 3 (pick_stages) returned an unusable response for capability %s; "
            "falling back to first candidate.",
            req.capability.value,
        )
        return [candidates[0].card.name]
    valid_names = {c.card.name for c in candidates}
    out: list[str] = []
    for name in chosen:
        if name in valid_names:
            out.append(name)
        else:
            logger.warning(
                "Step 3 (pick_stages) returned invalid stage %r for capability %s; "
                "dropping that pick.",
                name,
                req.capability.value,
            )
    if not out:
        logger.warning(
            "Step 3 (pick_stages) yielded zero valid stages for capability %s; "
            "falling back to first candidate.",
            req.capability.value,
        )
        return [candidates[0].card.name]
    return out


def _card_summary_for_picker(card: StageCard) -> dict[str, Any]:
    """A compact JSON-friendly view of a card that fits the Picker's context."""

    return {
        "name": card.name,
        "summary": card.summary,
        "capabilities": [t.value for t in card.capabilities],
        "also_handles": [t.value for t in card.also_handles],
        "cost_hint": card.cost_hint,
        "produces_cardinality": card.produces_cardinality.value,
        "selection_hints": {
            "prefer_when": card.selection_hints.prefer_when,
            "avoid_when": card.selection_hints.avoid_when,
            "notes": card.selection_hints.notes,
        },
    }


# ----------------------------------------------------------------------------
# Step 4: Param Tuner (LLM, one call per picked stage)
# ----------------------------------------------------------------------------


def tune_params(
    stage_names: list[str],
    intent: IntentCategories,
    registry: CapabilityRegistry,
    llm: LLMClient,
    *,
    tier: str = "synth",
) -> list[StageRef]:
    """Tune each picked stage's params from its threshold_bands / combo_presets.

    Per-stage focused LLM call. If the card has no tunable params we skip
    the LLM and emit a bare :class:`StageRef`.
    """

    sys_prompt = _load_prompt("tuner")
    out: list[StageRef] = []
    intent_dump = intent.model_dump(mode="json", exclude_none=True)
    for name in stage_names:
        entry = registry.get(name)
        if entry is None:
            logger.warning("Step 4 (tune_params): unknown stage %r — skipping", name)
            continue
        card = entry.card
        if not card.params:
            out.append(StageRef(stage=name, params={}))
            continue
        params = _llm_tune_one(card, intent_dump, llm, sys_prompt, tier=tier)
        out.append(StageRef(stage=name, params=params))
    return out


def _llm_tune_one(
    card: StageCard,
    intent_dump: dict[str, Any],
    llm: LLMClient,
    sys_prompt: str,
    *,
    tier: str,
) -> dict[str, Any]:
    payload = {
        "stage": card.name,
        "user_intent": intent_dump,
        "param_surface": [
            {
                "name": p.name,
                "type": p.type,
                "default": p.default,
                "min": p.min,
                "max": p.max,
                "choices": p.choices,
                "required": p.required,
                "description": p.description,
            }
            for p in card.params
        ],
        "threshold_bands": [b.model_dump(mode="json") for b in card.selection_hints.threshold_bands],
        "combo_presets": [c.model_dump(mode="json") for c in card.selection_hints.combo_presets],
    }
    messages = [
        Message("system", sys_prompt),
        Message("user", json.dumps(payload, indent=2)),
    ]
    raw = _call_json(llm, messages, tier=tier)
    params = raw.get("params", {})
    return _coerce_params_to_schema(card, params)


def _coerce_params_to_schema(card: StageCard, params: dict[str, Any]) -> dict[str, Any]:
    """Drop params that don't appear on the card; clamp numerics to min/max."""

    allowed = {p.name: p for p in card.params}
    cleaned: dict[str, Any] = {}
    for k, v in params.items():
        if k not in allowed:
            continue
        spec = allowed[k]
        if isinstance(v, (int, float)) and (spec.min is not None or spec.max is not None):
            lo = spec.min if spec.min is not None else float("-inf")
            hi = spec.max if spec.max is not None else float("inf")
            v = max(lo, min(hi, v))
        cleaned[k] = v
    return cleaned


# ----------------------------------------------------------------------------
# Step 5: validator — delegated to nemo_curator.agentic.validator.validate
# ----------------------------------------------------------------------------


def _validate(
    ir: PipelineIR,
    registry: CapabilityRegistry,
    intent: IntentCategories,
) -> ValidationReport:
    return validate(ir, registry, intent=intent, mutate=True)


# ----------------------------------------------------------------------------
# Step 6: Critic (LLM, bounded loop)
# ----------------------------------------------------------------------------


def critic_review(
    ir: PipelineIR,
    prompt: str,
    intent: IntentCategories,
    profile: DatasetCard | None,
    llm: LLMClient,
    *,
    tier: str = "synth",
) -> CriticVerdict:
    """Ask the LLM whether the assembled IR really answers the user's prompt.

    The critic only sees a compact summary, never the full prompt blob, so
    its token cost is bounded. Returns :class:`CriticVerdict`.
    """

    sys_prompt = _load_prompt("critic")
    payload = {
        "user_prompt": prompt,
        "intent": intent.model_dump(mode="json", exclude_none=True),
        "dataset_profile": profile.model_dump(mode="json") if profile else None,
        "pipeline": [
            {"stage": s.stage, "params": s.params, "auto_inserted": s.auto_inserted}
            for s in ir.stages
        ],
    }
    messages = [
        Message("system", sys_prompt),
        Message("user", json.dumps(payload, indent=2)),
    ]
    raw = _call_json(llm, messages, tier=tier)
    return CriticVerdict(
        approved=bool(raw.get("approved", False)),
        complaints=[str(c) for c in raw.get("complaints", [])],
        patch=raw.get("patch", {}) or {},
        score=raw.get("score"),
    )


# ----------------------------------------------------------------------------
# DAG entry point
# ----------------------------------------------------------------------------


def plan(
    prompt: str,
    source_uri: str,
    source_kind: str,
    target_dir: str,
    *,
    llm: LLMClient,
    registry: CapabilityRegistry,
    max_refine_iters: int = 2,
    sample_limit: int = 64,
    tier: str = "synth",
) -> PlannerResult:
    """Run the four-LLM-step DAG end-to-end.

    Steps 1-6 in order:

        1. extract_intent
        2. derive_capabilities
        3. pick_stages
        4. tune_params
        5. validate (auto-insert / topo-sort / key-flow)
        6. critic_review — if not approved AND max_refine_iters > 0, the
           critic's ``patch`` is applied (currently: drop or add stages by
           name) and Steps 3-5 are re-run.
    """

    # --- Step 0: dataset profile (deterministic) -------------------------
    source = SourceSpec(kind=source_kind, uri=source_uri)  # type: ignore[arg-type]
    profile: DatasetCard | None
    try:
        profile = profile_source(source, sample_limit=sample_limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Step 0 (profile_source) failed: %s — continuing without profile.", exc)
        profile = None

    # --- Step 1: intent extraction ---------------------------------------
    intent = extract_intent(prompt, profile, llm, tier=tier)

    # --- Steps 2..6 (with bounded refinement loop) -----------------------
    critic_history: list[CriticVerdict] = []
    iteration = 0
    while True:
        caps = derive_capabilities(intent)
        stage_names = pick_stages(caps, registry, llm, tier=tier)
        stages = tune_params(stage_names, intent, registry, llm, tier=tier)
        ir = PipelineIR(
            source=source,
            sink=SinkSpec(target_dir=target_dir),
            stages=stages,
            intent=intent,
        )
        report = _validate(ir, registry, intent)
        verdict = critic_review(report.ir, prompt, intent, profile, llm, tier=tier)
        critic_history.append(verdict)
        if verdict.approved or iteration >= max_refine_iters:
            return PlannerResult(
                ir=report.ir,
                report=report,
                intent=intent,
                profile=profile,
                critic_history=critic_history,
            )
        # Apply refinement patch -- currently we support "add" / "drop"
        # adjustments to the stage list. The next loop iteration re-runs
        # Steps 3-5 with the updated state.
        intent = _apply_refinement_patch(intent, verdict.patch)
        iteration += 1


def _apply_refinement_patch(intent: IntentCategories, patch: dict[str, Any]) -> IntentCategories:
    """Merge the critic's ``patch`` (intent-level only, for safety) into intent.

    The patch may contain a subset of ``IntentCategories`` field overrides.
    Anything else (stage add/drop) we accept silently for now and let the
    next planning iteration see the same intent — the critic's complaints
    will at least be visible in ``critic_history``.
    """

    if not patch:
        return intent
    intent_dump = intent.model_dump(mode="json", exclude_none=True)
    for k, v in patch.get("intent_overrides", {}).items():
        intent_dump[k] = v
    try:
        return IntentCategories.model_validate(intent_dump)
    except Exception as exc:  # noqa: BLE001
        logger.warning("refinement patch ignored — invalid: %s", exc)
        return intent


# ----------------------------------------------------------------------------
# Shared LLM helper
# ----------------------------------------------------------------------------


def _call_json(
    llm: LLMClient,
    messages: list[Message],
    *,
    tier: str,
    max_retries: int = 1,
    purpose: str = "planner_dag",
) -> dict[str, Any]:
    """LLM call that demands JSON output. One retry at temperature=0 on bad JSON."""

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return llm.chat_json(messages, tier=tier, temperature=0.0, purpose=purpose)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "planner DAG LLM call returned bad JSON (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
    raise ValueError(f"LLM returned non-JSON after {max_retries + 1} attempts: {last_exc}")


__all__ = [
    "CriticVerdict",
    "PlannerResult",
    "critic_review",
    "derive_capabilities",
    "extract_intent",
    "pick_stages",
    "plan",
    "tune_params",
]
