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

"""Pluggable validation-check registry for agent-composed audio recipes.

``verbs.validate`` builds a :class:`CheckContext`, runs every registered check,
and merges the results into a :class:`~nemo_curator.audio_agent.contracts.Verdict`.
Each check is ``fn(ctx) -> CheckResult``; adding a check is a ``@register`` + a
function, with no change to callers. This is the extension point for the 1B
correctness taxonomy (task-type, key-flow, output-completeness) — the checks are
added here, and the verb surface stays stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from nemo_curator.audio_agent.contracts import Issue
from nemo_curator.audio_agent.index import get_index

if TYPE_CHECKING:
    from nemo_curator.audio_agent.recipe import Recipe

# Params whose value is compared against a card ``max_speakers`` constraint.
_MAX_SPEAKERS_KEYS = ("max_speakers", "num_speakers")


@dataclass
class CheckContext:
    """Everything a validation check needs, assembled once by ``validate``."""

    recipe: Recipe
    stages: list[Any]  # built stage instances (well-formedness already passed)
    data_profile: dict[str, Any] | None
    env: Any  # EnvProfile instance (has .has_gpu / .has_ffmpeg / .available_secrets)
    initial_roles: set[str]
    initial_keys: set[str]
    available_gpus: float
    expected_outputs: list[str] = field(default_factory=list)  # roles the user asked for
    acceptance_criteria: list[Any] = field(default_factory=list)  # parsed AcceptanceCriterion list (1A.1)
    request_type: str | None = None  # goal/request kind, for request-type sanity (1A.1)


@dataclass
class CheckResult:
    """A single check's contribution, merged into the Verdict by ``run_checks``."""

    issues: list[Issue] = field(default_factory=list)
    card_violations: list[Issue] = field(default_factory=list)
    gate_flags: list[Issue] = field(default_factory=list)
    unproducible_roles: list[str] = field(default_factory=list)
    produced_roles: list[str] = field(default_factory=list)
    produced_keys: list[str] = field(default_factory=list)
    ok: bool | None = None
    keys_ok: bool | None = None


Check = Callable[[CheckContext], CheckResult]
REGISTRY: list[tuple[str, Check]] = []


def register(name: str) -> Callable[[Check], Check]:
    """Register a check under ``name`` (order of registration is run order)."""

    def deco(fn: Check) -> Check:
        REGISTRY.append((name, fn))
        return fn

    return deco


def _to_int(v: Any) -> int | None:  # noqa: ANN401
    """Best-effort int coercion (None on failure), so a malformed card value or an
    LLM-supplied param (e.g. num_speakers='two') can't raise out of a check."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def run_checks(ctx: CheckContext) -> CheckResult:
    """Run every registered check and merge the results into one ``CheckResult``.

    Each check is isolated: if one raises, it is converted into a ``check_error``
    issue (and the recipe is marked not-ok) instead of propagating out of
    ``validate`` -- the grounding layer must never emit a traceback where it promised
    a JSON Verdict.
    """
    merged = CheckResult()
    for name, fn in REGISTRY:
        try:
            r = fn(ctx)
        except Exception as e:  # noqa: BLE001 - a single check must never crash the verb
            merged.issues.append(
                Issue(
                    "check_error", "error",
                    f"check {name!r} could not run ({type(e).__name__}: {e}); recipe not fully validated",
                    fix="treat as not-runnable; fix the offending card/param, or report a bug",
                )
            )
            merged.ok = False
            continue
        merged.issues.extend(r.issues)
        merged.card_violations.extend(r.card_violations)
        merged.gate_flags.extend(r.gate_flags)
        merged.unproducible_roles.extend(r.unproducible_roles)
        if r.produced_roles:
            merged.produced_roles = r.produced_roles
        if r.produced_keys:
            merged.produced_keys = r.produced_keys
        if r.ok is not None:
            merged.ok = r.ok
        if r.keys_ok is not None:
            merged.keys_ok = r.keys_ok
    return merged


def _fix_for(code: str) -> str | None:
    return {
        "unsatisfied_reads": "insert an upstream stage that produces the missing role (see find_producers)",
        "dangling_key": "align the producer's *_key value with what this stage reads, or seed it from the source manifest",
        "tensor_into_sink": (
            "a resident tensor/audio blob is reaching a sink that serializes task.data as-is; strip it "
            "before the sink using a method that preserves the sink's input task type -- e.g. read/score "
            "from file (input_residency=file) or stop carrying the waveform (keep_waveform_in_task=false). "
            "A sanitizer stage only helps if its OUTPUT task type matches the sink (e.g. AudioToDocumentStage "
            "emits a DocumentBatch, so it fits a DocumentBatch sink, not an AudioTask sink like ManifestWriterStage)."
        ),
        "gpu_unavailable": "run on a GPU host or set the stage to CPU resources",
        "composite": "decompose the composite (it hides its true I/O) before validating downstream",
        "key_removed_upstream": "reorder so the reader runs before the stage that removes the key, or re-produce the key",
    }.get(code)


def _escalate_for(code: str) -> str | None:
    # A composite hides its writes, so an unsatisfied read past it is neither a hard
    # fail nor a clean pass -> mark the Verdict 'uncertain' (resolve it with a smoke).
    return {"unsatisfied_reads_after_composite": "smoke"}.get(code)


# --------------------------------------------------------------------------- #
# registered checks
# --------------------------------------------------------------------------- #
@register("data_flow")
def _check_data_flow(ctx: CheckContext) -> CheckResult:
    """Role/key composition, residency, serializability, GPU gate (foundation)."""
    from nemo_curator.stages.audio import agent as foundation

    report = foundation.validate_pipeline(
        ctx.stages,
        initial_roles=ctx.initial_roles,
        initial_keys=ctx.initial_keys,
        available_gpus=ctx.available_gpus,
    )
    issues = [
        Issue(pi.code, pi.severity, pi.message, stage_index=pi.stage_index, stage=pi.stage_name,
              fix=_fix_for(pi.code), escalate_to=_escalate_for(pi.code))
        for pi in report.issues
    ]
    return CheckResult(
        issues=issues,
        ok=report.ok,
        keys_ok=report.keys_ok,
        produced_roles=sorted(report.produced_roles),
        produced_keys=sorted(report.produced_keys),
    )


@register("card_constraints")
def _check_card_constraints(ctx: CheckContext) -> CheckResult:
    """Model-card constraints (batch, sample-rate, duration, max-speakers)."""
    idx = get_index()
    out: list[Issue] = []
    data_profile = ctx.data_profile
    # sample-rate keys are strings post-serialization; coerce to int so a matching rate
    # (16000) doesn't false-warn against an int-typed card supported_sample_rates.
    data_srs = {int(k) for k in (data_profile or {}).get("sample_rates", {}) if str(k).lstrip("-").isdigit()} if data_profile else set()
    mean_dur = float((data_profile or {}).get("mean_duration_sec", 0.0)) if data_profile else 0.0
    for i, s in enumerate(ctx.recipe.stages):
        card = idx.card(s.ref)
        if not card:
            continue
        cons = card.get("constraints", {}) or {}
        bs = cons.get("batch_size")
        if isinstance(bs, dict) and "fixed" in bs and s.params.get("batch_size") not in (None, bs["fixed"]):
            out.append(
                Issue(
                    "card_batch_size", "error",
                    f"{s.ref}: batch_size must be {bs['fixed']} ({bs.get('reason', 'model constraint')})",
                    stage_index=i, stage=s.ref, fix=f"set batch_size={bs['fixed']}",
                )
            )
        supported = cons.get("supported_sample_rates")
        supported_ints = {iv for iv in (_to_int(x) for x in (supported or [])) if iv is not None}
        if supported and data_srs and supported_ints and not data_srs.issubset(supported_ints):
            out.append(
                Issue(
                    "card_sample_rate", "warning",
                    f"{s.ref}: input sample rates {sorted(data_srs)} not all in supported {supported}",
                    stage_index=i, stage=s.ref, fix="insert a resample/mono stage upstream to the supported rate",
                )
            )
        sweet = cons.get("input_duration_sweetspot_sec")
        if isinstance(sweet, dict) and sweet.get("max") and mean_dur and mean_dur > float(sweet["max"]):
            out.append(
                Issue(
                    "card_duration", "warning",
                    f"{s.ref}: mean input duration {mean_dur}s exceeds sweet-spot max {sweet['max']}s",
                    stage_index=i, stage=s.ref, fix="segment/split long audio upstream",
                )
            )
        for key in _MAX_SPEAKERS_KEYS:
            mx_int = _to_int(cons.get("max_speakers"))
            val_int = _to_int(s.params.get(key))
            if mx_int is not None and val_int is not None and val_int > mx_int:
                out.append(
                    Issue(
                        "card_max_speakers", "error",
                        f"{s.ref}: {key}={s.params[key]} exceeds model max_speakers={mx_int}",
                        stage_index=i, stage=s.ref, fix=f"set {key}<={mx_int}",
                    )
                )
    return CheckResult(card_violations=out)


@register("gates")
def _check_gates(ctx: CheckContext) -> CheckResult:
    """Environment gates: ffmpeg / GPU / first-run download / runtime secrets."""
    from nemo_curator.stages.audio import agent as foundation

    env = ctx.env
    out: list[Issue] = []
    for idx, st in enumerate(ctx.stages):
        try:
            gates = foundation.build_contract(st).gates
        except Exception:  # noqa: BLE001
            continue
        name = type(st).__name__
        if getattr(gates, "requires_ffmpeg", False) and not env.has_ffmpeg:
            out.append(Issue("ffmpeg_missing", "error", f"{name} needs ffmpeg but it is not on PATH", stage_index=idx, stage=name, fix="install ffmpeg"))
        if getattr(gates, "requires_gpu", False) and not env.has_gpu:
            out.append(Issue("gpu_unavailable", "warning", f"{name} declares requires_gpu but no GPU was detected", stage_index=idx, stage=name, fix="run on a GPU host or lower resources"))
        if getattr(gates, "requires_internet_first_run", False):
            out.append(Issue("internet_first_run", "info", f"{name} downloads a model on first run", stage_index=idx, stage=name))
        for secret in getattr(gates, "runtime_secrets", []) or []:
            if secret not in env.available_secrets:
                out.append(Issue("missing_secret", "warning", f"{name} needs secret {secret!r} which is not set", stage_index=idx, stage=name, fix=f"export {secret}"))
    return CheckResult(gate_flags=out)


@register("unproducible")
def _check_unproducible(ctx: CheckContext) -> CheckResult:
    """Roles the pipeline reads that no stage in the catalog can produce."""
    from nemo_curator.stages.audio import agent as foundation

    required: set[str] = set()
    for st in ctx.stages:
        try:
            c = foundation.build_contract(st)
        except Exception:  # noqa: BLE001
            continue
        for key in [*c.reads.data_keys, *c.reads.segment_data_keys]:
            required.add(c.key_roles.get(key, "unknown"))
    return CheckResult(unproducible_roles=get_index().unproducible(sorted(required - {"unknown"})))


@register("output_completeness")
def _check_output_completeness(ctx: CheckContext) -> CheckResult:
    """Every requested output role must be produced by some stage in the recipe.

    Only active when the caller passes ``expected_outputs`` (semantic roles the
    user asked for). Catches the "asked for transcripts, no ASR stage" class.
    Phase 2 compiles ``expected_outputs`` from ``GoalSpec.acceptance_criteria``.
    """
    if not ctx.expected_outputs:
        return CheckResult()
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio._conformance import produced_roles

    available_roles = set(ctx.initial_roles)
    available_keys = set(ctx.initial_keys)
    for st in ctx.stages:
        try:
            contract = foundation.build_contract(st)
        except Exception:  # noqa: BLE001
            continue
        available_roles |= produced_roles(contract)
        available_keys |= set(contract.writes.data_keys) | set(contract.writes.segment_data_keys)
    out: list[Issue] = []
    for want in ctx.expected_outputs:
        # Satisfied by a produced semantic role OR a literal produced key. The key match
        # lets output-completeness distinguish specific metrics (e.g. wer vs a SIGMOS
        # sub-score) that all share the generic "score" role.
        if want not in available_roles and want not in available_keys:
            out.append(
                Issue(
                    "missing_output_producer", "error",
                    f"requested output {want!r} is not produced by any stage in the recipe (no matching role or key)",
                    fix="add a stage that produces this output (see discover / find_producers), or drop the requirement",
                )
            )
    return CheckResult(issues=out)


@register("request_type_sanity")
def _check_request_type_sanity(ctx: CheckContext) -> CheckResult:
    """Acceptance-set sanity (1A.1): a request implying an output must carry the
    matching criterion (filtering -> yield; transcription -> output_completeness).

    Output-completeness itself is enforced by the ``output_completeness`` check
    (``validate`` compiles criterion fields into ``expected_outputs``); this check
    only surfaces a *missing implied criterion* so success can't be declared while
    silently ignoring the point of the request. Warning + escalate-to-user (not a
    hard fail: the user may legitimately omit it, but it's flagged at the gate).
    Inactive unless a ``request_type`` or criteria were supplied.
    """
    if not ctx.request_type and not ctx.acceptance_criteria:
        return CheckResult()
    from nemo_curator.audio_agent.acceptance import missing_implied

    out = [
        Issue(
            "missing_implied_criterion", "warning", hint,
            fix="add an acceptance criterion of this type to define success for the request",
            escalate_to="user",
        )
        for _implied, hint in missing_implied(ctx.request_type, ctx.acceptance_criteria)
    ]
    return CheckResult(issues=out)


@register("task_type")
def _check_task_type(ctx: CheckContext) -> CheckResult:
    """Consecutive stages must be task-type compatible.

    A ``DocumentBatch`` producer (e.g. ``AudioToDocumentStage``) feeding an
    ``AudioTask``-only stage is the ``AudioToDocument -> ManifestWriter`` bug
    class. Types are auto-derived from the ``ProcessingStage[X, Y]`` generic;
    a boundary where a type can't be derived is skipped (no false mismatch).
    """
    from nemo_curator.stages.audio import agent as foundation

    contracts: list[Any] = []
    for st in ctx.stages:
        try:
            contracts.append(foundation.build_contract(st))
        except Exception:  # noqa: BLE001
            contracts.append(None)
    out: list[Issue] = []
    for i in range(len(contracts) - 1):
        up, dn = contracts[i], contracts[i + 1]
        if up is None or dn is None:
            continue
        prod, acc = up.produces_task_type, dn.accepts_task_type
        if prod and acc and prod != acc:
            up_name, dn_name = type(ctx.stages[i]).__name__, type(ctx.stages[i + 1]).__name__
            out.append(
                Issue(
                    "task_type_mismatch", "error",
                    f"{up_name} produces {prod} but {dn_name} accepts {acc}",
                    stage_index=i + 1, stage=dn_name,
                    fix="insert a converter (e.g. AudioToDocumentStage) or reorder so task types line up",
                )
            )
    return CheckResult(issues=out)


# Diarization/speaker-separation needs a continuous waveform. If a fragmenting VAD
# stage precedes a diarizer/separator with no re-join in between, the diarizer gets a
# torn, per-segment signal - the enforced `diarization-needs-continuous-audio` rule
# (patterns/composition.yaml). Separating BEFORE the VAD (on continuous audio) is fine.
# All three roles are derived from card metadata so a NEW stage is covered with no code
# change, unioned with an explicit fallback for a card that is missing/miscategorized:
#   diarizer   = card category 'diarize'
#   fragmenter = card category 'segment' + a 'fanout' tag (a VAD-style per-segment split),
#                excluding diarizers (SpeakerSeparation also fans out but is a diarizer)
#   re-joiner  = the stage that stitches segments back into a continuous waveform; the only
#                such stage is SegmentConcatenationStage and there is no distinct card signal
#                for it, so it stays an explicit set (wrapped in a helper for symmetry).
_DIARIZERS = frozenset({"InferenceSortformerStage", "PyAnnoteDiarizationStage", "SpeakerSeparationStage"})
_FRAGMENTERS = frozenset({"VADSegmentationStage", "WhisperXVADStage"})
_REJOINERS = frozenset({"SegmentConcatenationStage"})


def _is_diarizer(ref: str, idx: Any) -> bool:  # noqa: ANN401
    """A diarizer/separator: card category 'diarize' (extensible) or the explicit set."""
    if ref in _DIARIZERS:
        return True
    return (idx.card(ref) or {}).get("category") == "diarize"


def _is_fragmenter(ref: str, idx: Any) -> bool:  # noqa: ANN401
    """A VAD-style stage that fans continuous audio into per-segment tasks (breaking
    continuity): the explicit set, or any card with category 'segment' AND a 'fanout' tag
    that is not itself a diarizer -- so a NEW VAD-style stage is covered with no code change.
    """
    if ref in _FRAGMENTERS:
        return True
    card = idx.card(ref) or {}
    return card.get("category") == "segment" and "fanout" in (card.get("tags") or []) and not _is_diarizer(ref, idx)


def _is_rejoiner(ref: str, idx: Any) -> bool:  # noqa: ANN401
    """A stage that stitches segments back into a continuous waveform. Structural: the only
    such stage today is SegmentConcatenationStage, so this is an explicit set kept behind a
    helper for symmetry with the derived diarizer/fragmenter checks (and a future card signal).
    """
    return ref in _REJOINERS


@register("diarization_continuity")
def _check_diarization_continuity(ctx: CheckContext) -> CheckResult:
    """Flag a diarizer/separator that runs after VAD fragmentation without a
    SegmentConcatenation re-join (it would see per-segment clips, not continuous
    audio). Diarizing/separating on the continuous audio before any VAD does not trip.
    """
    idx = get_index()
    refs = [s.ref for s in ctx.recipe.stages]
    out: list[Issue] = []
    for di, r in enumerate(refs):
        if not _is_diarizer(r, idx):
            continue
        frags_before = [fi for fi, fr in enumerate(refs[:di]) if _is_fragmenter(fr, idx)]
        if not frags_before:
            continue  # runs on continuous audio (no upstream fragmentation) -> fine
        fi = max(frags_before)  # nearest fragmenter before the diarizer
        if not any(_is_rejoiner(rr, idx) for rr in refs[fi + 1:di]):
            out.append(
                Issue(
                    "diarization_needs_continuous_audio", "error",
                    f"{r} runs after {refs[fi]} without re-joining segments; "
                    "diarization/separation needs a continuous waveform",
                    stage_index=di, stage=r,
                    fix="insert SegmentConcatenationStage between the VAD and the diarizer, "
                        "or diarize/separate on the continuous audio before segmenting",
                )
            )
    return CheckResult(issues=out)
