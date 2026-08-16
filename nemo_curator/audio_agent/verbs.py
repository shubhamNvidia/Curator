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

"""The deterministic verb surface the host LLM (and CLI/MCP) drives.

Each verb returns a JSON-safe dict. Retrieval/validation/execution/reporting is
ours (deterministic, grounded); interpret/route/plan/critique is the host's.

    discover / describe / catalog_tree / cards / context   -> knowledge + routing
    validate                                               -> Verdict (grounds the plan)
    smoke                                                  -> bounded evidence
    run                                                    -> confirm-gated full run + report
    report                                                 -> post-hoc evidence from outputs
"""

from __future__ import annotations

import contextlib
import copy
import filecmp
import functools
import ipaddress
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from nemo_curator.audio_agent import _safety
from nemo_curator.audio_agent import context as _context
from nemo_curator.audio_agent.contracts import DATASET_KEY_TIERS, Issue, SmokeReport, Verdict
from nemo_curator.audio_agent.index import get_index
from nemo_curator.audio_agent.profiler import probe_env, profile_data
from nemo_curator.audio_agent.recipe import (
    EXECUTION_KNOB_PARAMS,
    OUTPUT_LOCATION_PARAMS,
    Recipe,
    build_stages,
)
from nemo_curator.audio_agent.report import (
    _row_count,
    build_run_report,
    rows_written_in,
    sparse_fields_in,
    stage_duration_sec,
)

# How many output rows a success contract may read back as evidence. Enough to judge coverage
# on a real corpus; small enough that verifying a finished run never becomes the expensive part.
_EVIDENCE_ROWS = 2000

# Smoke runs exercise real write behavior, but only inside an ephemeral tree.
# The ``raw_data_dir`` subtraction is defensive rather than active: that param is not in
# ``OUTPUT_LOCATION_PARAMS`` today, so it removes nothing. It stays because if the set ever
# grows to cover it, redirecting it would be wrong -- for a dataset source it is execution
# INPUT as well as an acquisition destination, and smoke refuses generated sources outright
# instead (``_bound_recipe``), keeping pre-staged roots read-only.
_SMOKE_FILE_OUTPUT_PARAMS = frozenset(
    {"output_path", "output_manifest", "output_audio_tar_path"}
)
_SMOKE_DIR_OUTPUT_PARAMS = frozenset(
    OUTPUT_LOCATION_PARAMS
    - _SMOKE_FILE_OUTPUT_PARAMS
    - {"raw_data_dir"}
)



# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ray_address_is_local(address: str | None) -> bool:
    """Whether a configured Ray address names this machine.

    An unset address plus Ray's ``auto`` and ``local`` modes retain local checks. A head
    THIS process bootstrapped is always local: it runs on this machine even when Ray
    advertises the node's LAN IP (multi-NIC / cloud), so its probed VRAM/GPU names are
    valid. Otherwise a concrete address is local only when its host is ``localhost`` or an
    IPv4/IPv6 loopback address; unknown hostnames fail closed as remote so driver
    environment facts are never projected onto another machine.
    """
    value = str(address or "").strip()
    if not value or value.casefold() in {"auto", "local"}:
        return True
    try:  # a self-started head is on this machine regardless of the advertised IP
        from nemo_curator.audio_agent import _ray

        if _ray.owns_cluster(value):
            return True
    except Exception:  # noqa: BLE001 - ownership is best-effort; fall through to host checks
        pass
    if value.casefold().rstrip(".") in {"localhost", "::1", "[::1]"}:
        return True
    try:
        parsed = urlsplit(value if "://" in value else f"//{value}")
        host = (parsed.hostname or "").casefold().rstrip(".")
    except ValueError:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _ray_execution_target(address: str | None) -> str:
    """Map a Ray address to the environment-analysis target vocabulary."""
    return "local" if _ray_address_is_local(address) else "external_ray"


def _as_recipe(recipe: Recipe | dict[str, Any]) -> Recipe:
    rec = recipe if isinstance(recipe, Recipe) else Recipe.from_dict(recipe)
    # ``Recipe(...)`` remains a convenient SDK constructor and can bypass
    # Recipe.from_dict. Hold that path to the same fail-closed contract boundary
    # before validate/smoke/run/reuse performs any I/O.
    from nemo_curator.audio_agent.acceptance import parse_criteria

    parse_criteria(rec.acceptance_criteria)
    return rec


def _dataset_binding(rec: Recipe, data: str | None):  # noqa: ANN202 - private adapter type
    """Resolve the recipe's physical input; stage params remain execution truth."""
    from nemo_curator.audio_agent.input_identity import resolve_dataset_binding

    return resolve_dataset_binding(rec, data)


def _binding_blocks_execution(binding: Any, data: str | None) -> bool:  # noqa: ANN401
    """Whether a binding problem makes execution unsafe rather than merely unkeyed."""
    if binding.status in {"missing", "mismatch", "unsupported"}:
        return True
    if binding.status != "ambiguous":
        return False
    # A configured multi-manifest reader can still execute exactly as authored
    # when no singular --data assertion pretends to describe the whole set. It
    # simply gets no reuse identity until aggregate profiling exists.
    return not (
        data is None
        and binding.source_ref == "ManifestReader"
        and binding.source_index == 0
        and len(binding.configured_paths) > 1
    )


def _profile_binding(binding: Any):  # noqa: ANN202, ANN401 - DataProfile | None without eager imports
    """Profile only the source the recipe actually executes, never caller text."""
    if binding.status not in {"resolved", "ambiguous"} or not binding.profile_source:
        return None
    allowed = {
        "audio_filepath_key",
        "audio_dir",
        "audio_path_resolution",
        "case_sensitive_extensions",
        "exclude_stage_intermediates",
        "folder_extensions",
        "identity_files",
        "max_files",
        "max_probe",
        "recursive",
    }
    kwargs = {key: value for key, value in binding.profile_kwargs.items() if key in allowed}
    prof = profile_data(binding.profile_source, **kwargs)
    # A lexical remote selector or unsupported selector must not become a
    # reusable shape key merely because DataProfile can hash its spelling.
    return prof if prof.kind != "unknown" else None


def _profile_error(profile: Any) -> str:  # noqa: ANN401
    """Fatal source-definition error found by the read-only profiler."""
    errors = list(getattr(profile, "source_errors", None) or [])
    return "; ".join(str(error) for error in errors)


def _profile_refusal(binding: Any, profile: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "status": "refused",
        "reason": f"recipe data source is unreadable or malformed: {_profile_error(profile)}",
        "data_binding": binding.to_dict(),
        "data_profile": profile.to_dict(),
    }


def _binding_issue(binding: Any, data: str | None) -> Issue | None:  # noqa: ANN401
    """Translate source binding state into the validation issue vocabulary."""
    if binding.status == "resolved":
        return None
    blocking = _binding_blocks_execution(binding, data)
    code = {
        "missing": "data_source_missing",
        "mismatch": "data_source_mismatch",
        "ambiguous": "data_source_ambiguous",
        "unsupported": "data_source_unsupported",
    }.get(binding.status, "data_source_unresolved")
    return Issue(
        code,
        "error" if blocking else "warning",
        binding.reason,
        stage_index=binding.source_index,
        stage=binding.source_ref,
        fix={
            "missing": "configure an existing source in the first supported source stage",
            "mismatch": (
                "make --data and Recipe.inputs agree with the first stage's configured source"
            ),
            "unsupported": (
                "use a supported first source stage/path form or add an explicit source adapter"
            ),
            "ambiguous": (
                "remove the extra source, or omit singular --data for an authored "
                "multi-manifest reader"
                if blocking
                else "omit singular --data or use one manifest selector if reusable identity is required"
            ),
        }.get(binding.status, "bind one supported source as the recipe's first stage"),
    )


def _binding_refusal(binding: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "status": "refused",
        "reason": f"recipe data source is not safely bound: {binding.reason}",
        "data_binding": binding.to_dict(),
    }


@dataclass(frozen=True)
class _ContinuationRunContext:
    """The minimum claim a continued run may present at the execution boundary.

    None of the identities that are ultimately published are accepted directly.
    The verifier re-derives the logical chain and the materialized suffix from
    this recipe, the current source fingerprint, and one complete artifact.
    """

    logical_recipe: dict[str, Any]
    dataset_key: str
    reuse_step_key: str


@dataclass(frozen=True)
class _PretrainFinalizer:
    """Driver lifecycle required by the shard-writing ALM pretrain stages."""

    manifest_path: str
    metrics_path: str
    audio_tar_path: str
    audio_filepath_key: str
    audio_tar_expected: bool = True

    def prepare(self) -> None:
        from nemo_curator.stages.audio.alm.pretrain import (
            prepare_audio_pretrain_outputs,
        )

        prepare_audio_pretrain_outputs(
            self.manifest_path,
            self.metrics_path,
            self.audio_tar_path,
        )

    def finalize(self, *, successful: bool = True) -> int:
        """Merge this attempt's shards and return its serialized snippet count.

        A successful attempt owns the final output paths, including the truthful
        empty-output case.  A failed attempt may still have recoverable shards,
        but must not erase an older completed output when it produced none.
        """
        from nemo_curator.stages.audio.alm.pretrain import (
            finalize_audio_pretrain_outputs,
        )

        finalize_audio_pretrain_outputs(
            self.manifest_path,
            self.metrics_path,
            self.audio_tar_path,
            audio_filepath_key=self.audio_filepath_key,
            replace_empty=successful,
            audio_tar_expected=self.audio_tar_expected,
        )
        return _count_output_rows(self.manifest_path)


def _pretrain_finalizer(
    stages: list[Any],
) -> tuple[_PretrainFinalizer | None, str]:
    """Resolve a complete ALM shard lifecycle, or reject a partial recipe."""
    wanted: dict[str, list[Any]] = {
        "SnippetExtractionStage": [],
        "SnippetManifestWriterStage": [],
        "PretrainMetricsAggregatorStage": [],
    }
    for stage in stages:
        name = type(stage).__name__
        if name in wanted:
            wanted[name].append(stage)
    present = {name for name, matches in wanted.items() if matches}
    if not present:
        return None, ""
    if present != set(wanted):
        missing = sorted(set(wanted) - present)
        return None, (
            "ALM pretrain shard outputs require SnippetExtractionStage, "
            "SnippetManifestWriterStage, and PretrainMetricsAggregatorStage "
            f"in one recipe; missing {missing}"
        )
    duplicates = sorted(
        name for name, matches in wanted.items() if len(matches) != 1
    )
    if duplicates:
        return None, (
            "ALM pretrain finalization requires exactly one of each shard stage; "
            f"ambiguous stage(s): {duplicates}"
        )

    extraction = wanted["SnippetExtractionStage"][0]
    writer = wanted["SnippetManifestWriterStage"][0]
    aggregator = wanted["PretrainMetricsAggregatorStage"][0]
    manifest_path = getattr(writer, "output_path", None)
    metrics_path = getattr(aggregator, "output_path", None)
    audio_tar_path = getattr(extraction, "output_audio_tar_path", None)
    missing_paths = sorted(
        key
        for key, value in {
            "manifest_path": manifest_path,
            "metrics_path": metrics_path,
            "audio_tar_path": audio_tar_path,
        }.items()
        if not isinstance(value, str) or not value
    )
    if missing_paths:
        return None, (
            "ALM pretrain finalization is missing required output path(s): "
            f"{missing_paths}"
        )
    return (
        _PretrainFinalizer(
            manifest_path=str(manifest_path),
            metrics_path=str(metrics_path),
            audio_tar_path=str(audio_tar_path),
            audio_filepath_key=str(
                getattr(extraction, "audio_filepath_key", "audio_filepath")
            ),
            audio_tar_expected=not bool(getattr(extraction, "dry_run", False)),
        ),
        "",
    )


@dataclass(frozen=True)
class _SmokeBound:
    """A proved smoke-input cap, or a refusal before stage construction."""

    recipe: Recipe | None
    tmp_paths: tuple[str, ...] = ()
    input_count: int | None = None
    error: str = ""
    output_root: str | None = None


def _continuation_context_from_plan(
    logical_recipe: Recipe,
    plan: dict[str, Any],
) -> _ContinuationRunContext:
    """Snapshot the logical request for verification by :func:`run`."""
    point = plan.get("reuse_point") or {}
    return _ContinuationRunContext(
        logical_recipe=logical_recipe.to_dict(),
        dataset_key=str(plan.get("dataset_key") or ""),
        reuse_step_key=str(point.get("step_key") or ""),
    )


def _verify_continuation_context(  # noqa: PLR0911 - each refusal names one broken proof
    physical_recipe: Recipe,
    physical_binding: Any,  # noqa: ANN401 - DatasetBinding without eager import
    context: _ContinuationRunContext | None,
) -> tuple[dict[str, Any] | None, str]:
    """Prove a continued run, then derive every identity override internally.

    A caller cannot provide step tuples or a run-record chain. The reused step
    must belong to the logical recipe on the stated current dataset; its complete
    artifact must be the physical source; and materializing that exact boundary
    must reproduce the recipe about to execute.
    """
    if context is None:
        return None, ""
    if not isinstance(context, _ContinuationRunContext):
        return None, "continued-run context has an invalid type"

    from nemo_curator.audio_agent import artifacts as art_mod
    from nemo_curator.audio_agent import continuation as continuation_mod
    from nemo_curator.audio_agent.input_identity import canonical_source

    dataset_key = str(context.dataset_key or "")
    reuse_step_key = str(context.reuse_step_key or "")
    if not dataset_key or not reuse_step_key:
        return None, "continued-run context is incomplete"

    try:
        logical_recipe = Recipe.from_dict(context.logical_recipe).freeze()
        logical_binding = _dataset_binding(logical_recipe, None)
        logical_profile = _profile_binding(logical_binding)
    except (TypeError, ValueError) as exc:
        return None, f"continued-run logical recipe is invalid: {exc}"
    if logical_binding.status != "resolved" or logical_profile is None:
        return None, "continued-run logical source cannot be verified"
    if logical_profile.dataset_key() != dataset_key:
        return None, "continued-run logical source no longer matches the claimed dataset"

    try:
        logical_plans = art_mod.plan_steps(logical_recipe, dataset_key)
    except Exception as exc:  # noqa: BLE001 - turn an identity failure into a refusal
        return None, f"continued-run logical steps could not be derived: {exc}"
    matches = [plan for plan in logical_plans if plan.step_key == reuse_step_key]
    if len(matches) != 1:
        return None, "continued-run artifact step does not belong to the logical recipe"
    reused_plan = matches[0]

    artifact, reasons = art_mod.lookup(reuse_step_key, dataset_key=dataset_key)
    if artifact is None or reasons:
        why = "; ".join(reasons or ["artifact record is missing"])
        return None, f"continued-run lineage is no longer valid: {why}"
    expected_identity = {
        "step_key": reused_plan.step_key,
        "dataset_key": dataset_key,
        "input_key": reused_plan.input_key,
        "stage_index": reused_plan.index,
        "stage_ref": reused_plan.stage_ref,
        "semantic_params": reused_plan.semantic_params,
        "model_version": reused_plan.model_version,
    }
    try:
        actual_identity = {
            "step_key": str(getattr(artifact, "step_key", "") or ""),
            "dataset_key": str(getattr(artifact, "dataset_key", "") or ""),
            "input_key": str(getattr(artifact, "input_key", "") or ""),
            "stage_index": int(getattr(artifact, "stage_index", -1)),
            "stage_ref": str(getattr(artifact, "stage_ref", "") or ""),
            "semantic_params": dict(getattr(artifact, "semantic_params", {}) or {}),
            "model_version": str(getattr(artifact, "model_version", "") or ""),
        }
    except (TypeError, ValueError):
        return None, "continued-run artifact metadata is malformed"
    if actual_identity != expected_identity:
        return None, "continued-run artifact metadata does not match its logical step"

    if not physical_binding.primary_path:
        return None, "continued-run physical source could not be resolved"
    try:
        if canonical_source(physical_binding.primary_path) != canonical_source(artifact.uri):
            return None, "continued-run recipe does not read the verified artifact"
    except (TypeError, ValueError) as exc:
        return None, f"continued-run artifact path is invalid: {exc}"

    expected_recipe, materialize_error = continuation_mod.materialize(
        logical_recipe,
        uri=artifact.uri,
        kind=artifact.kind,
        prefix=reused_plan.index + 1,
    )
    if expected_recipe is None:
        return None, f"continued-run boundary cannot be materialized: {materialize_error}"
    if physical_recipe.compute_hash() != expected_recipe.compute_hash():
        return None, "continued-run physical recipe is not the verified logical suffix"

    return {
        "dataset_key": dataset_key,
        "fingerprint_tier": str(getattr(artifact, "fingerprint_tier", "") or ""),
        "data_source": logical_binding.primary_path,
        "step_identity": [
            (plan.step_key, plan.input_key, plan.index)
            for plan in logical_plans[reused_plan.index :]
        ],
        "logical_steps": [plan.step_key for plan in logical_plans],
    }, ""


def _derive_initial(data_profile: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    """Seed initial roles + literal keys from the input data profile."""
    from nemo_curator.stages.audio._roles import role_for_value

    keys: set[str] = {"audio_filepath"}
    if data_profile and data_profile.get("manifest_keys"):
        keys |= set(data_profile["manifest_keys"])
    roles = {role_for_value(k) for k in keys} | {"audio_filepath"}
    roles.discard("unknown")
    return roles, keys


# --------------------------------------------------------------------------- #
# discovery + routing (L0/L1/L2)
# --------------------------------------------------------------------------- #
def discover() -> dict[str, Any]:
    """List every agent-ready audio stage with its category + one-liner.

    Also reports stage modules that failed to import (``unavailable``), so a shorter
    catalog is never mistaken for a smaller library: on a supported CPU-only install the
    ASR/diarization stages are absent, and the host must be able to say "unavailable here,
    install the GPU extra" instead of "this cannot be done". The key is omitted entirely
    when everything imported, so a healthy environment's output is unchanged.
    """
    from nemo_curator.stages.audio._catalog import unavailable_modules

    idx = get_index()
    stages = [
        {"stage": name, "category": idx.category_of(name), "summary": idx.one_liner(name), "tags": idx.tags_of(name)}
        for name in idx.stage_names()
    ]
    out: dict[str, Any] = {"count": len(stages), "stages": stages}
    unavailable = unavailable_modules()
    if unavailable:
        out["unavailable"] = unavailable
    return out


def describe(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the contract (+ card, if any) for one stage, resolved against ``params``.

    Pass the params the recipe will use. Reads and writes follow from them --
    ``SplitLongAudioStage(segments_key="diar_segments")`` reads ``diar_segments`` -- so a
    contract asked for without them describes the defaults, not the pipeline. Params a stage
    does not accept, or required ones left out, produce a labelled fallback naming what to
    supply rather than a contract that appears to require nothing.
    """
    from nemo_curator.audio_agent._resolve import resolved_contract_for

    out: dict[str, Any] = {"stage": name, "category": get_index().category_of(name)}
    try:
        resolved = resolved_contract_for(name, params)
    except KeyError:
        return {"stage": name, "error": f"{name!r} is not a registered agent-ready audio stage"}
    except Exception as e:  # noqa: BLE001 - e.g. an optional dependency failing to import
        out["contract_error"] = f"{type(e).__name__}: {e}"
    else:
        out["contract"] = resolved.contract.to_dict()
        detail = resolved.unresolved_detail()
        if detail is not None:
            out["contract_unresolved"] = detail
        inner = _composite_detail(resolved.instance)
        if inner is not None:
            out["expands_to"] = inner
        varies = _contract_variants(name, resolved, params)
        if varies:
            out["contract_varies_with"] = varies
    card = get_index().card(name)
    if card:
        out["card"] = card
    return out


def _contract_variants(
    name: str,
    resolved: Any,  # noqa: ANN401 - ResolvedContract
    params: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Which other settings of an enumerable param would change what this stage reads or writes.

    A resolved contract answers for ONE configuration, and the caller cannot tell from it whether
    that was the only possibility. ``input_residency`` is the case that matters: at its ``file``
    default a stage reads ``audio_filepath``, at ``waveform`` it reads ``waveform`` +
    ``sample_rate``, and at ``auto`` it takes either. A caller holding an in-task waveform and
    reading only the default contract concludes the stage cannot consume it, and either inserts a
    needless write-to-disk or abandons the stage.

    Derived by resolving each declared choice and keeping only those whose reads or writes
    actually differ, so this reports facts rather than restating the parameter list -- and stays
    empty for the 37 stages with nothing enumerable to say. Every alternative costs one more
    constructor call, which is why this can be exhaustive: 53 across the whole catalog.
    """
    instance = resolved.instance
    if instance is None:
        return []
    from nemo_curator.audio_agent._resolve import resolved_contract_for

    baseline = _io_view(resolved.contract)
    out: list[dict[str, Any]] = []
    for spec in resolved.contract.params:
        if not spec.choices:
            continue
        current = getattr(instance, spec.name, spec.default)
        differing: list[dict[str, Any]] = []
        for choice in spec.choices:
            if choice == current:
                continue
            trial = dict(params or {})
            trial[spec.name] = choice
            candidate = resolved_contract_for(name, trial)
            if not candidate.resolved:
                continue
            view = _io_view(candidate.contract)
            if view != baseline:
                differing.append({"value": choice, **view})
        if differing:
            out.append({"param": spec.name, "current": current, "changes_it_to": differing})
    return out


def _composite_detail(instance: Any) -> dict[str, Any] | None:  # noqa: ANN401 - any configured stage
    """What a composite requires and produces, from the stages it actually expands into.

    A composite describes only itself: ``SplitASRAlignJoinStage.describe()`` declares no reads
    and no writes, so its resolved contract is as empty as the unresolved one and for a different
    reason. The work -- and the requirements -- live in ``decompose()``.

    ``requires_upstream`` is the actionable part: each inner read that no earlier inner stage
    produces, which is what the caller has to arrange before the composite will start. For a
    ``SplitASRAlignJoinStage`` that is ``segments``, the requirement that a pipeline producing
    ``diar_segments`` failed to meet after two model downloads and a GPU diarization pass.

    Uses the expansion primitive that pipeline validation and resource planning already share, so
    a composite cannot report one shape here and a different one when it runs.
    """
    if instance is None:
        return None
    from nemo_curator.stages.audio._composite import expand_composites

    expansion = expand_composites([instance])
    leaves = [item for item in expansion.stages if item.stage is not instance]
    if not leaves:
        # Two different answers, and only one of them means "we could not tell". A stage the
        # executor will refuse outright contributes no leaf either, and reporting that as
        # silence -- or as mere opacity -- is the same "empty is not an answer" failure this
        # whole verb exists to end: the caller would read no requirements and conclude there
        # are none, for a stage that cannot run at all.
        unrunnable = expansion.unrunnable.get(0)
        if unrunnable:
            return {"unrunnable": unrunnable}
        reason = expansion.opaque.get(0)
        return {"opaque": reason} if reason else None

    from nemo_curator.stages.audio._agent_registry import build_contract

    produced: set[str] = set()
    required: list[str] = []
    alternatives: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for item in leaves:
        try:
            contract = build_contract(item.stage)
        except Exception as e:  # noqa: BLE001 - one unreadable child does not hide its siblings
            stages.append({"stage": type(item.stage).__name__, "error": f"{type(e).__name__}: {e}"})
            continue
        entry: dict[str, Any] = {"stage": type(item.stage).__name__, **_io_view(contract)}
        stages.append(entry)
        required.extend(k for k in contract.reads.data_keys if k not in produced and k not in required)
        unmet = _unmet_alternatives(contract, produced)
        if unmet:
            alternatives.append({"stage": entry["stage"], "one_of": unmet})
        produced.update(contract.writes.data_keys)

    out: dict[str, Any] = {"stages": stages, "requires_upstream": required, "produces": sorted(produced)}
    if alternatives:
        # Kept separate from ``requires_upstream`` rather than flattened into it. A stage reading
        # audio accepts a file path OR an in-task waveform; listing both as required would demand
        # a form the caller does not need, and picking one for them would hide the other -- the
        # same "an empty field is not an answer" failure in a different costume.
        out["requires_one_of"] = alternatives
    return out


def _io_view(contract: Any) -> dict[str, Any]:  # noqa: ANN401 - StageContract
    """The keys a contract reads and writes, including alternatives it would accept."""
    view: dict[str, Any] = {
        "reads": list(contract.reads.data_keys),
        "writes": list(contract.writes.data_keys),
    }
    one_of = [list(spec.data_keys) for spec in contract.reads_one_of if spec.data_keys]
    if one_of:
        view["reads_one_of"] = one_of
    return view


def _unmet_alternatives(contract: Any, produced: set[str]) -> list[list[str]]:  # noqa: ANN401
    """Alternative read-sets when none of them is already satisfied, else nothing.

    Many stages declare no flat ``reads`` at all and put every requirement here --
    ``BandwidthEstimationStage`` reads ``audio_filepath`` plus one of ``segments``/``duration``
    through this field alone. Consulting only ``reads.data_keys`` would report such a stage as
    needing nothing from upstream.
    """
    options = [list(spec.data_keys) for spec in contract.reads_one_of if spec.data_keys]
    if not options or any(set(option) <= produced for option in options):
        return []
    return options


def producers(role: str) -> dict[str, Any]:
    """Which stages write ``role`` -- by semantic role, or by literal key name.

    Answers the question that otherwise sends a caller into the source tree: a stage requires
    ``segments`` and nothing says who makes one. Matching accepts either vocabulary because a
    caller reading a contract has a key name in hand and a caller reading a request has a role,
    and making them guess which to ask with is how the question gets abandoned.

    Built from contracts resolved at DEFAULT params, so it answers "which stage can produce
    this" rather than "which stage will, as configured". A stage whose output key is renamed
    through a param still appears under its default key; ``describe`` with the real params is
    the authority on any specific recipe.
    """
    from nemo_curator.audio_agent._resolve import resolved_contract_for
    from nemo_curator.stages.audio._roles import role_for_value

    wanted = str(role)
    proven: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    not_searched: list[str] = []

    for entry in discover().get("stages", []):
        stage = str(entry.get("stage") or "")
        category = entry.get("category")
        try:
            resolved = resolved_contract_for(stage)
        except Exception as e:  # noqa: BLE001 - an unimportable stage is reported, not fatal
            not_searched.append(f"{stage} ({type(e).__name__})")
            continue
        if resolved.resolved:
            hit = _writes_role(resolved.contract, wanted, role_for_value)
            if hit:
                proven.append({"stage": stage, "category": category, **hit})
            continue
        # Unresolvable at defaults, so its writes are unknown -- but its params still DECLARE
        # the roles of the keys it names, and that declaration is data rather than a guess.
        # Kept apart from ``producers`` because a declared role does not distinguish a key the
        # stage reads from one it writes: enough to point the caller at ``describe``, not enough
        # to answer for it. Placeholder values were the alternative and they fabricate keys --
        # probing PreserveByValueStage that way has it "writing" a key called `placeholder`.
        if wanted in set(resolved.contract.key_roles.values()):
            candidate: dict[str, Any] = {
                "stage": stage,
                "category": category,
                "matched": "declared_role",
                "confirm_with": f"describe({stage!r}, params={{...}})",
                "reason": resolved.unresolved_reason,
            }
            if resolved.required_params:
                candidate["needs_params"] = list(resolved.required_params)
            candidates.append(candidate)
        else:
            not_searched.append(stage)

    out: dict[str, Any] = {"role": wanted, "producers": proven}
    if candidates:
        out["candidates"] = candidates
    if not_searched:
        # "Nothing produces this" and "some stages could not be asked" are different answers and
        # only one of them means stop looking, so an incomplete search never presents as complete.
        out["not_searched"] = sorted(not_searched)
    return out


def _writes_role(
    contract: Any,  # noqa: ANN401 - StageContract, imported lazily by callers
    wanted: str,
    role_for_value: Any,  # noqa: ANN401
) -> dict[str, Any] | None:
    """How ``contract`` produces ``wanted``, or ``None`` if it does not."""
    writes = getattr(contract, "writes", None)
    keys = list(getattr(writes, "data_keys", []) or [])
    segment_keys = list(getattr(writes, "segment_data_keys", []) or [])
    key_roles = dict(getattr(contract, "key_roles", {}) or {})

    if wanted in keys:
        return {"writes_key": wanted, "matched": "key"}
    if wanted in segment_keys:
        return {"writes_segment_key": wanted, "matched": "segment_key"}
    for key in keys + segment_keys:
        if (key_roles.get(key) or role_for_value(key)) == wanted:
            return {"writes_key": key, "matched": "role"}
    return None


def catalog_tree() -> dict[str, Any]:
    """L0: the full category tree the host prunes over before drilling in."""
    return {"categories": get_index().category_tree()}


def cards(category: str | None = None, names: list[str] | None = None) -> dict[str, Any]:
    """L1 (one-liners for a category) or L2 (full cards for named finalists)."""
    idx = get_index()
    if names:
        return {"cards": idx.full_cards(names)}
    if category:
        return {"category": category, "stages": idx.card_oneliners(category)}
    msg = "cards() requires either a category (L1) or a list of names (L2)"
    raise ValueError(msg)


def context(
    goal: dict[str, Any] | None = None,
    *,
    data: str | None = None,
    stages: list[str] | None = None,
    roles: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a compact PlanningContext for the host router/planner.

    Unlike recipe-driven verbs, ``data`` is the direct path to profile because
    no executable recipe source exists yet.
    """
    pviol = _safety.path_violations([data])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    return _context.assemble(goal, data=data, selected_stages=stages, roles=roles).to_dict()


def resolve(
    stage: str,
    *,
    label: str | None = None,
    use_case: str | None = None,
    explicit: dict[str, Any] | None = None,
    data: str | None = None,
) -> dict[str, Any]:
    """Resolve an outcome to concrete stage config via the card (1A.2).

    Path A maps a user-facing outcome — a quality ``label`` ("studio"), a named
    ``use_case`` preset, or an ``explicit`` ``{param: value}`` — to concrete
    params (or, for an annotator, a ``PreserveByValueStage`` filter), plus an
    auditable ``strategy`` trail. Keeps internal thresholds out of user questions
    and never invents a number.

    Path B (passing ``data``) adds the values the DATASET fixes rather than the user —
    currently the rate the audio actually is. It configures the stage for this recipe and
    never changes a stage default, so hand-written pipelines and tutorials are unaffected.
    Ambiguity comes back as an ``ask``: mixed sample rates are a question for the user
    rather than something to guess at.

    Supplying ``data`` IS the request for Path B. It used to need a separate
    ``data_driven=True`` alongside it, from when data context was optional; every other
    data-taking verb now resolves and profiles its source on its own, and the second flag
    only ever meant that a caller who passed one without the other silently got nothing.

    It does NOT infer which column holds the audio. ``audio_filepath`` is the NeMo manifest
    key, and supplying a manifest in that format is the caller's responsibility; a source
    that does not carry it is refused by the ``source_schema`` check with the columns it
    does have, rather than guessed at. Guessing would mean guessing wrong on some dataset
    and silently curating the wrong field.
    """
    from nemo_curator.audio_agent import config_strategy

    result = config_strategy.resolve(
        stage, label=label, use_case=use_case, explicit=explicit
    )
    if not data:
        return result
    # Held to the same workspace lock as every other verb that is handed a dataset path.
    # This one profiles ``data`` off the filesystem too, and was the only such verb without
    # the check -- unreachable while no adapter passed ``data``, and a hole the moment one did.
    pviol = _safety.path_violations([data])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    profile: dict[str, Any] | None = None
    with contextlib.suppress(Exception):  # a source we cannot profile simply adds nothing
        profile = profile_data(data).to_dict()
    # Path A wins on conflict: an outcome the user asked for outranks an inference. Telling
    # Path B what is already decided, rather than merging over it afterwards, keeps the
    # ``strategy`` trail honest -- it used to record the data-informed value and its rationale
    # for a param the merge then discarded.
    derived = config_strategy.resolve_from_data(
        stage, profile, already_set=set(result["params"])
    )
    result["params"] = {**derived["params"], **result["params"]}
    result["strategy"] = list(result["strategy"]) + list(derived["strategy"])
    asks = list(result["asks"]) + list(derived["asks"])
    if derived["params"] or derived["asks"]:
        # Path B answered (or asked something concrete), so Path A's "you gave me no
        # outcome" prompt is no longer true and would only add noise.
        asks = [a for a in asks if a != config_strategy.NO_OUTCOME_ASK]
    result["asks"] = asks
    return result


def diagnose(
    error: str,
    *,
    recipe: Recipe | dict[str, Any] | None = None,
    operation: str = "run",
    phase: str = "runtime",
    attempted_actions: list[str] | None = None,
    execution_target: str | None = None,
) -> dict[str, Any]:
    """Analyze a captured failure for the host LLM without applying a fix.

    The result combines the sanitized failure taxonomy, a fresh machine probe,
    recipe-specific environment applicability, ranked grounded choices, and an
    explicit user-decision prompt. Unknown failures stay unknown and receive only
    diagnostic next steps.
    """
    stages: list[Any] = []
    recipe_issues: list[dict[str, Any]] = []
    if recipe is not None:
        stages_or_none, recipe_issues = build_stages(_as_recipe(recipe))
        stages = list(stages_or_none or [])
    target = (
        execution_target
        if execution_target in {"local", "external_ray", "custom_executor"}
        else _ray_execution_target(os.environ.get("RAY_ADDRESS"))
    )
    from nemo_curator.audio_agent.diagnostics import diagnose_failure

    result = diagnose_failure(
        error,
        stages=stages,
        env=probe_env(),
        operation=operation,
        phase=phase,
        execution_target=target,
        attempted_actions=attempted_actions,
    )
    if recipe_issues:
        result["recipe_issues"] = recipe_issues
    return _safety.redact(result)


# --------------------------------------------------------------------------- #
# validate (mechanical composition + card facts + preflight)
# --------------------------------------------------------------------------- #
def validate(
    recipe: Recipe | dict[str, Any],
    *,
    data: str | None = None,
    initial_keys: list[str] | None = None,
    initial_roles: list[str] | None = None,
    expected_outputs: list[str] | None = None,
    acceptance_criteria: list[dict[str, Any]] | None = None,
    request_type: str | None = None,
) -> dict[str, Any]:
    """Validate a recipe: does it compose, and can it run in this environment?

    Well-formedness is checked here (real stages, constructible params); the rest
    runs through the pluggable check registry (``audio_agent.checks``): data-flow
    (role/key/residency/serialization), card constraints, environment gates,
    unproducible roles, task-type, output-completeness, and request-type sanity.
    ``expected_outputs`` (semantic roles) enables the output-completeness check;
    ``acceptance_criteria`` (1A.1) additionally compile their output/metric fields
    into that check and drive request-type sanity via ``request_type``. They are
    a cross-check only: the same success contract must be embedded in the recipe
    so smoke/run and the confirmation hash cannot lose it.

    Mechanical progression requires returned ``runnable`` (no error-severity
    problem anywhere) or ``status == "pass"`` -- NOT ``ok``, which is data-flow
    only. Before smoke or confirmation, the host must also complete the returned
    ``semantic_review`` and reach an intent-level pass.

    ``output_targets`` states what is already at each location the recipe writes to, so an
    occupied output can be REPORTED to the user rather than guessed at or quietly cleared.

    ``semantic_review`` is a separate, advisory evidence packet for the host LLM.
    It co-locates configured field lineage, cardinality seams, and card prose but
    never claims that the recipe expresses the user's intent.

    The first supported source stage's parameters are execution truth. ``data``
    and ``Recipe.inputs`` are optional assertions and never rewrite that stage.
    """
    from nemo_curator.audio_agent.acceptance import expected_roles_from_criteria, parse_criteria
    from nemo_curator.audio_agent.checks import CheckContext, run_checks

    rec = _as_recipe(recipe)
    verdict = Verdict()

    if not rec.stages:
        verdict.issues.append(Issue("empty_recipe", "error", "recipe has no stages"))
        return verdict.to_dict()

    pviol = _safety.path_violations([data, *_safety.recipe_path_params(rec)])
    if pviol:
        verdict.issues.append(
            Issue(
                "path_outside_workspace",
                "error",
                "; ".join(pviol),
                fix="move the source/output under AUDIO_AGENT_WORKSPACE or change that lock",
            )
        )
        return verdict.to_dict()

    binding = _dataset_binding(rec, data)
    verdict.data_binding = binding.to_dict()
    binding_issue = _binding_issue(binding, data)
    if binding_issue is not None:
        verdict.issues.append(binding_issue)
    dp_obj = _profile_binding(binding)
    data_profile = dp_obj.to_dict() if dp_obj else None
    if dp_obj is not None and _profile_error(dp_obj):
        verdict.issues.append(
            Issue(
                "data_source_unreadable",
                "error",
                _profile_error(dp_obj),
                stage_index=binding.source_index,
                stage=binding.source_ref,
                fix="repair or replace the manifest before validation or execution",
            )
        )
    env = probe_env()
    execution_target = _ray_execution_target(os.environ.get("RAY_ADDRESS"))

    stages, build_issues = build_stages(rec)
    verdict.issues.extend(_issue_from_dict(i) for i in build_issues)
    if stages is None:
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        verdict.diagnosis = diagnose_failure(
            "; ".join(
                str(issue.get("message") or issue.get("code") or "")
                for issue in build_issues
            ),
            env=env,
            operation="validate",
            phase="stage_construction",
            execution_target=execution_target,
        )
        return verdict.to_dict()

    roles0, keys0 = _derive_initial(data_profile)
    if initial_roles is not None:
        roles0 = set(initial_roles)
    if initial_keys is not None:
        keys0 = set(initial_keys)

    recipe_criteria = parse_criteria(rec.acceptance_criteria)
    if acceptance_criteria is None:
        criteria = recipe_criteria
    else:
        criteria = parse_criteria(acceptance_criteria)
        if criteria != recipe_criteria:
            verdict.issues.append(
                Issue(
                    "acceptance_contract_not_embedded",
                    "error",
                    "the criteria passed to validate differ from the recipe's "
                    "acceptance_criteria, so a later smoke/run would not be bound "
                    "to the success contract that was validated",
                    fix=(
                        "copy the complete acceptance_criteria list into recipe.yaml "
                        "and validate that same recipe"
                    ),
                )
            )
    expected = set(expected_outputs or []) | set(expected_roles_from_criteria(criteria))

    ctx = CheckContext(
        recipe=rec,
        stages=stages,
        data_profile=data_profile,
        env=env,
        initial_roles=roles0,
        initial_keys=keys0,
        available_gpus=float(env.gpu_count) if env.has_gpu else 0.0,
        expected_outputs=sorted(expected),
        acceptance_criteria=criteria,
        request_type=request_type,
        execution_target=execution_target,
    )
    result = run_checks(ctx)
    verdict.ok = bool(result.ok)
    verdict.keys_ok = bool(result.keys_ok)
    verdict.produced_roles = result.produced_roles
    verdict.produced_keys = result.produced_keys
    verdict.issues.extend(result.issues)
    verdict.card_violations.extend(result.card_violations)
    verdict.gate_flags.extend(result.gate_flags)
    verdict.unproducible_roles = result.unproducible_roles

    # Mechanical validation cannot decide whether a valid field/filter/model
    # expresses open-ended user intent.  Assemble the exact configured lineage
    # and its card prose for the mandatory host-LLM critic, without letting an
    # advisory packet failure take down the deterministic validator.
    semantic_response: dict[str, Any] | None = None
    try:
        from nemo_curator.audio_agent.semantic_review import (
            build_semantic_review,
            semantic_response_contract,
        )

        semantic_response = semantic_response_contract()
        verdict.semantic_review = build_semantic_review(
            stages,
            initial_keys=keys0,
            recipe=rec,
            data_profile=data_profile,
        )
    except Exception as exc:  # noqa: BLE001 - validation stays available; packet says it is incomplete
        verdict.semantic_review = {
            "status": "unavailable",
            "review_required": True,
            "advisory_only": True,
            "intent_interpretation_performed": False,
            "required_response": semantic_response,
            "contract_issues": [
                {
                    "code": "semantic_review_context_unavailable",
                    "message": _safety.redact_secret_text(
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            ],
            "checklist": [
                {
                    "id": "missing_review_context",
                    "required": True,
                    "instruction": (
                        "Do not infer intent from a mechanically runnable verdict; "
                        "retrieve the configured stage cards and lineage manually."
                    ),
                }
            ],
        }

    from nemo_curator.audio_agent.diagnostics import (
        environment_preflight,
        verdict_issues,
    )

    environment_decision = environment_preflight(
        stages,
        env,
        operation="validate",
        execution_target=execution_target,
    )
    verdict.environment_decision = environment_decision
    for env_issue in verdict_issues(environment_decision):
        matches = [
            issue
            for issue in verdict.gate_flags
            if issue.code == env_issue.code
        ]
        if matches:
            # Keep the existing per-stage location, but promote a proven,
            # recipe-relevant execution blocker to the shared safe severity.
            for issue in matches:
                issue.severity = "error"
                issue.fix = env_issue.fix
                issue.escalate_to = "user"
        else:
            verdict.gate_flags.append(env_issue)
    verdict.output_targets = _output_targets(rec)
    return verdict.to_dict()


def _output_targets(rec: Recipe) -> list[dict[str, Any]]:
    """What is already sitting at each location this recipe writes to.

    Facts only, no predictions: whether it exists, whether it is a file or a directory, and how
    much is in it. An agent that cannot see this has to guess, and guessing led somewhere bad --
    reading the writer's append-mode open, concluding reruns would double, and deleting the
    user's file before the confirm gate. The path was in fact replaced cleanly by the run.

    Reporting an occupied path is the whole point; deciding what to do about it belongs to the
    user, and clearing it belongs to nobody (the pipeline replaces its own manifest output).
    """
    targets: list[dict[str, Any]] = []
    for path in _recipe_outputs(rec, None):
        expanded = os.path.expanduser(path)
        entry: dict[str, Any] = {"path": path, "exists": os.path.exists(expanded)}
        if not entry["exists"]:
            targets.append(entry)
            continue
        if os.path.isdir(expanded):
            entry["kind"] = "directory"
            with contextlib.suppress(OSError):
                entry["files"] = sum(len(fs) for _r, _d, fs in os.walk(expanded))
        else:
            entry["kind"] = "file"
            with contextlib.suppress(OSError):
                entry["bytes"] = os.path.getsize(expanded)
            if expanded.endswith((".jsonl", ".json")):
                entry["rows"] = _count_output_rows(expanded)
        entry["note"] = "already present; the run writes here. Report this -- do not clear it yourself."
        targets.append(entry)
    return targets


def _issue_from_dict(d: dict[str, Any]) -> Issue:
    return Issue(
        code=d.get("code", "error"),
        severity=d.get("severity", "error"),
        message=d.get("message", ""),
        stage_index=d.get("stage_index"),
        stage=d.get("stage"),
        fix=d.get("fix"),
    )


# --------------------------------------------------------------------------- #
# smoke + run + report
# --------------------------------------------------------------------------- #
def _is_streaming_infeasible(err: Exception) -> bool:
    m = str(err).lower()
    return "streaming mode" in m and any(k in m for k in ("not enough gpu", "batch mode", "requires"))


def _run_pipeline_autofallback(
    stages: list[Any], mode: str, caller_executor: Any, *,  # noqa: ANN401
    checkpoint_path: str | None = None, note: Any = None,  # noqa: ANN401
    before_retry: Any = None,  # noqa: ANN401
) -> tuple[list[Any] | None, str]:
    """Run in the planned mode; on a runtime streaming-infeasibility error (e.g. a
    composite stage hides inner GPU stages, so the planner undercounted the concurrent
    GPU reservation), retry once in batch -- Xenna's own recommended remedy.
    """
    executor = caller_executor if caller_executor is not None else _make_executor(mode)
    try:
        return _run_pipeline(stages, executor, checkpoint_path=checkpoint_path), mode
    except Exception as e:  # noqa: BLE001
        if caller_executor is None and mode != "batch" and _is_streaming_infeasible(e):
            if callable(note):
                note("streaming infeasible at runtime -> retried in batch")
            if callable(before_retry):
                # Reset hook for writers whose output lifecycle is NOT the standard
                # truncate-in-setup() (e.g. the ALM pretrain finalizer's prepare/finalize):
                # clear their partial state so the two attempts are never merged. Standard
                # terminal sinks (ManifestWriterStage / DocumentBatchJsonlWriterStage)
                # truncate their output in setup(), which the batch retry re-invokes, so
                # their partial streaming shards are REPLACED (not merged) without this hook
                # -- which is why before_retry is None for a plain writer pipeline.
                before_retry()
            return _run_pipeline(stages, _make_executor("batch"), checkpoint_path=checkpoint_path), "batch"
        raise


def _stops_a_head_it_started(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Guarantee that a Ray head bootstrapped by this call is stopped if the call raises.

    ``run`` and ``smoke`` stop their own head on every path they ANTICIPATE: a planning
    error, an infeasible plan, a normal return. What they cannot cover is the long tail
    between execution and that final stop -- re-binding the dataset, publishing artifacts,
    assembling the report -- which is not inside any handler. An unexpected error there
    escaped past the stop and left a live local head, with its ownership record and temp
    directory, behind in a process that had already given up on the work. The next call then
    found ``RAY_ADDRESS`` pointing at that head and refused as ambiguous, so one unrelated
    failure made every subsequent run refuse until the process was restarted.

    Ownership is module state in ``_ray``, so what needs stopping is knowable from outside
    the verb: anything owned on the way out that was not owned on the way in was started
    here. A successful return has already stopped and cleared it, so this finds nothing to
    do -- which is why it can wrap the verb whole instead of re-indenting its body.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        from nemo_curator.audio_agent._ray import owns_cluster, shutdown_cluster

        owned_before = owns_cluster()
        try:
            return fn(*args, **kwargs)
        except BaseException:  # KeyboardInterrupt too: Ctrl-C must not orphan a live head
            if not owned_before and owns_cluster():
                # No address argument: ownership is the whole question, and it has just been
                # answered. ``shutdown_cluster`` re-checks it and stops what this process
                # owns, with the same refusal gates as every other teardown path.
                with contextlib.suppress(Exception):
                    shutdown_cluster()
            raise

    return wrapper


@_stops_a_head_it_started
def smoke(
    recipe: Recipe | dict[str, Any],
    *,
    sample: int = 10,
    data: str | None = None,
    executor: Any = None,  # noqa: ANN401
    output_dir: str | None = None,
    bootstrap_ray: bool = False,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the recipe on a bounded sample and return structured evidence.

    ``bootstrap_ray`` opts into auto-starting a correctly-configured local Ray
    head when none is reachable (see ``_ray.ensure_cluster``). The result carries a
    ``calibration`` block — measured per-stage resources extracted from this run
    (1C.2) — to feed the next ``run``/``smoke`` planner; an optional ``calibration``
    input seeds mode selection from a prior smoke.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. A mismatch is refused before execution.
    """
    rec = _as_recipe(recipe).freeze()
    if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
        return {
            "status": "refused",
            "reason": "smoke sample must be a positive integer",
            "sample": sample,
            "config_hash": rec.config_hash,
        }
    pviol = _safety.path_violations([data, output_dir, *_safety.recipe_path_params(rec)])
    if pviol:
        return {"status": "refused", "reason": "path(s) resolve outside the allowed workspace", "violations": pviol}
    binding = _dataset_binding(rec, data)
    if _binding_blocks_execution(binding, data):
        return _safety.redact(_binding_refusal(binding))
    dp_obj = _profile_binding(binding)
    if dp_obj is not None and _profile_error(dp_obj):
        return _safety.redact(_profile_refusal(binding, dp_obj))
    preflight_stages, preflight_build_issues = build_stages(rec)
    execution_target = (
        "custom_executor"
        if executor is not None
        else _ray_execution_target(os.environ.get("RAY_ADDRESS"))
    )
    preflight_env = probe_env()
    if preflight_stages is None:
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        build_error = "; ".join(
            str(issue.get("message") or issue.get("code") or "")
            for issue in preflight_build_issues
        )
        return _safety.redact(
            {
                "status": "error",
                "reason": "smoke recipe could not be constructed",
                "issues": preflight_build_issues,
                "diagnosis": diagnose_failure(
                    build_error,
                    env=preflight_env,
                    operation="smoke",
                    phase="stage_construction",
                    execution_target=execution_target,
                ),
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    from nemo_curator.audio_agent.diagnostics import environment_preflight

    environment_decision = environment_preflight(
        preflight_stages,
        preflight_env,
        operation="smoke",
        execution_target=execution_target,
    )
    if not environment_decision.get("can_execute", False):
        return _safety.redact(
            {
                "status": "refused",
                "reason_code": "environment_action_required",
                "reason": environment_decision.get("summary"),
                "environment_decision": environment_decision,
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    rpt = SmokeReport(sample=sample)
    bound = _bound_recipe(rec, sample, rpt, binding)
    if bound.recipe is not None:
        bound = _isolate_smoke_outputs(bound, rpt)
    if bound.recipe is None:
        _cleanup(list(bound.tmp_paths))
        return _safety.redact(
            {
                "status": "refused",
                "reason": f"smoke input cannot be safely bounded: {bound.error}",
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    bounded = bound.recipe
    tmp_paths = list(bound.tmp_paths)

    stages, issues = build_stages(bounded)
    if stages is None:
        rpt.errors.extend(i.get("message", "") for i in issues)
        _cleanup(tmp_paths)
        out = rpt.to_dict()
        out["status"] = "error"
        out["reason"] = "smoke recipe could not be constructed"
        out["config_hash"] = rec.config_hash
        out["data_binding"] = binding.to_dict()
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        out["diagnosis"] = diagnose_failure(
            "; ".join(rpt.errors),
            env=preflight_env,
            operation="smoke",
            phase="bounded_stage_construction",
            execution_target=execution_target,
        )
        return _safety.redact(out)
    try:
        write_issues = _smoke_write_issues(stages, bound.output_root)
    except Exception as exc:  # noqa: BLE001 - output proof must fail closed
        write_issues = [
            "disk-write contracts could not be fully inspected "
            f"({type(exc).__name__}: {exc})"
        ]
    if write_issues:
        _cleanup(tmp_paths)
        return _safety.redact(
            {
                "status": "refused",
                "reason": "smoke output isolation could not be proven",
                "issues": write_issues,
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    pretrain_finalizer, pretrain_error = _pretrain_finalizer(stages)
    if pretrain_error:
        _cleanup(tmp_paths)
        return _safety.redact(
            {
                "status": "refused",
                "reason": pretrain_error,
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )

    data_profile = dp_obj.to_dict() if dp_obj else None
    profile_count = int((data_profile or {}).get("num_files", 0))
    rpt.input_count = (
        bound.input_count
        if bound.input_count is not None
        else min(sample, profile_count)
        if data_profile is not None
        else sample
    )
    caller_executor = executor
    owned_ray_address: str | None = None
    ray_address = (
        os.environ.get("RAY_ADDRESS")
        if caller_executor is None
        else None
    )
    if bootstrap_ray and caller_executor is None:
        try:
            from nemo_curator.audio_agent._ray import owns_cluster

            owned_before = owns_cluster()
            ray_address = _bootstrap_ray()
            if not owned_before and owns_cluster(ray_address):
                owned_ray_address = ray_address
            rpt.notes.append("ray_head=" + ray_address)
        except Exception as exc:  # noqa: BLE001 - structured preflight failure
            _cleanup(tmp_paths)
            from nemo_curator.audio_agent.diagnostics import diagnose_failure

            error_text = f"{type(exc).__name__}: {exc}"
            return _safety.redact(
                {
                    "status": "error",
                    "reason": (
                        "Ray bootstrap failed before resource planning: "
                        + error_text
                    ),
                    "diagnosis": diagnose_failure(
                        error_text,
                        stages=stages,
                        env=preflight_env,
                        operation="smoke",
                        phase="ray_bootstrap",
                        execution_target=execution_target,
                    ),
                    "sample": sample,
                    "config_hash": rec.config_hash,
                    "data_binding": binding.to_dict(),
                }
            )
    try:
        env_obj = _resource_environment(
            preflight_env,
            ray_address,
            execution_target,
        )
        rplan = _plan_resources(
            stages,
            env_obj,
            data_profile,
            calibration=calibration,
        )
        _adapt_resource_plan_for_target(
            rplan,
            execution_target=execution_target,
            operation="smoke",
        )
    except Exception as exc:  # noqa: BLE001 - structured failure + guaranteed cleanup
        ray_cleanup = _shutdown_owned_ray(owned_ray_address)
        _cleanup(tmp_paths)
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        error_text = f"{type(exc).__name__}: {exc}"
        return _safety.redact(
            {
                "status": "error",
                "reason": (
                    "smoke resource planning failed: "
                    + error_text
                ),
                "diagnosis": diagnose_failure(
                    error_text,
                    stages=stages,
                    env=preflight_env,
                    operation="smoke",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
                **(
                    {"ray_bootstrap_cleanup": ray_cleanup}
                    if ray_cleanup is not None
                    else {}
                ),
            }
        )
    rpt.notes.append(f"resource_plan_mode={rplan.mode}")
    if rplan.escalations:
        rpt.notes.append("resource_escalations=" + "; ".join(rplan.escalations))
    if not getattr(rplan, "feasible", True):
        ray_cleanup = _shutdown_owned_ray(owned_ray_address)
        _cleanup(tmp_paths)
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        infeasible = "; ".join(rplan.escalations) or "resource plan infeasible"
        return _safety.redact(
            {
                "status": "refused",
                "reason": "smoke resource plan is infeasible on the selected execution target",
                "sample": sample,
                "config_hash": rec.config_hash,
                "escalations": list(rplan.escalations),
                "machine_plan": rplan.to_dict(),
                "diagnosis": diagnose_failure(
                    infeasible,
                    stages=stages,
                    env=env_obj,
                    operation="smoke",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "data_binding": binding.to_dict(),
                **(
                    {"ray_bootstrap_cleanup": ray_cleanup}
                    if ray_cleanup is not None
                    else {}
                ),
            }
        )
    t0 = time.perf_counter()
    pretrain_prepared = False
    pretrain_output_rows: int | None = None
    runtime_diagnosis: dict[str, Any] | None = None
    try:
        if pretrain_finalizer is not None:
            pretrain_finalizer.prepare()
            pretrain_prepared = True
            rpt.notes.append("alm_pretrain_prepare=completed")
        results, _used_mode = _run_pipeline_autofallback(
            stages,
            rplan.mode,
            caller_executor,
            note=rpt.notes.append,
            before_retry=(
                pretrain_finalizer.prepare
                if pretrain_finalizer is not None
                else None
            ),
        )
        rpt.ran = True
        if pretrain_finalizer is not None:
            finalized_rows = pretrain_finalizer.finalize()
            if isinstance(finalized_rows, int) and not isinstance(finalized_rows, bool):
                pretrain_output_rows = max(0, finalized_rows)
            rpt.notes.append("alm_pretrain_finalize=completed")
        # ALM returns one origin stub when every snippet was filtered.  Only its
        # manifest is a serialized output, so count that instead of return
        # carriers or a zero-output smoke could receive a success token.
        rpt.retained = (
            pretrain_output_rows
            if pretrain_output_rows is not None
            else _row_count(results)
        )
        rpt.rejected = max(0, min(sample, rpt.input_count) - rpt.retained)
        rpt.per_stage_metrics = _stage_metrics(results)
        sampled_rows = list(_result_rows(results))
        rpt.examples = _examples_from_rows(sampled_rows, limit=3)
        rpt.goals_met = rpt.retained > 0
        incomplete_outputs = _empty_required_outputs(rec, sampled_rows)
        if incomplete_outputs:
            # A stage may retain rows while omitting a required field or filling it
            # with no content (e.g. an ASR that emits blank transcripts).
            # retained>0 alone would wrongly read as success.
            rpt.goals_met = False
            rpt.notes.append(
                "required output(s) MISSING or EMPTY in sampled rows: "
                + ", ".join(incomplete_outputs)
            )
    except Exception as e:  # noqa: BLE001 - classify any execution failure for the critic
        if pretrain_finalizer is not None and pretrain_prepared:
            try:
                pretrain_finalizer.finalize(successful=False)
                rpt.notes.append("alm_pretrain_partial_finalize=completed")
            except Exception as finalize_exc:  # noqa: BLE001 - retain the primary failure
                rpt.notes.append(
                    "alm_pretrain_partial_finalize=failed: "
                    f"{type(finalize_exc).__name__}: {finalize_exc}"
                )
        error_text = f"{type(e).__name__}: {e}"
        rpt.errors.append(error_text)
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        runtime_diagnosis = diagnose_failure(
            error_text,
            stages=stages,
            env=env_obj,
            operation="smoke",
            phase="pipeline_execution",
            execution_target=execution_target,
        )
        rpt.notes.append(
            "failure_code="
            + str((runtime_diagnosis.get("failure") or {}).get("code") or "unknown_failure")
        )
    finally:
        rpt.notes.append(f"elapsed_sec={round(time.perf_counter() - t0, 3)}")
        ray_cleanup = _shutdown_owned_ray(owned_ray_address)
        if ray_cleanup is not None:
            rpt.notes.append(
                "ray_bootstrap_cleanup="
                + ("completed" if ray_cleanup else "failed")
            )
        _cleanup(tmp_paths)
    out = rpt.to_dict()
    out["status"] = "completed" if rpt.ran and not rpt.errors else "error"
    out["config_hash"] = rec.config_hash
    out["data_binding"] = binding.to_dict()
    out["environment_decision"] = environment_decision
    if runtime_diagnosis is not None:
        out["diagnosis"] = runtime_diagnosis
    from nemo_curator.audio_agent import calibration as _cal

    out["calibration"] = _cal.from_smoke(out, machine_fingerprint=rplan.machine_fingerprint)
    # Keep the measurements where the next run can find them, so a plan is informed whether or
    # not the caller remembers --calibration. Meeting the sampled goals is deliberately NOT a
    # condition: what a stage used is a fact about resources, independent of whether the
    # sampled data satisfied the recipe.
    if rpt.ran and not rpt.errors and out["calibration"]:
        from nemo_curator.audio_agent import calibration_store

        out["calibration_stored"] = bool(
            calibration_store.save(
                rec.config_hash,
                out["calibration"],
                machine_fingerprint=rplan.machine_fingerprint,
            )
        )
    redacted = _safety.redact(out)
    if output_dir:
        redacted["warnings"] = [
            "output_dir is a legacy no-op; smoke outputs are always redirected "
            "to an ephemeral sandbox. Configure output paths on recipe stages."
        ]
    # Surface the smoke token AFTER redaction: it is evidence the host must hand back
    # to run() (AUDIO_AGENT_REQUIRE_SMOKE), not a secret to hide from the host. redact()
    # would otherwise strip it because the key contains "token".
    if rpt.ran and rpt.goals_met is True and not rpt.errors:
        redacted["smoke_token"] = _safety.smoke_token(rec.config_hash)
    else:
        redacted["smoke_token_status"] = (
            "not_issued: smoke must run without errors and meet sampled goals"
        )
    return redacted


@_stops_a_head_it_started
def run(  # noqa: PLR0913 - one verb, one keyword per execution knob (kept flat on purpose)
    recipe: Recipe | dict[str, Any],
    *,
    confirm: bool | str = False,
    data: str | None = None,
    executor: Any = None,  # noqa: ANN401
    output_dir: str | None = None,
    checkpoint_path: str | None = None,
    bootstrap_ray: bool = False,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
    goal: dict[str, Any] | None = None,
    reuse: dict[str, Any] | None = None,
    _continuation_context: _ContinuationRunContext | None = None,
) -> dict[str, Any]:
    """Confirm-gated full run. Refuses without explicit confirmation (0 silent runs).

    ``confirm`` may be ``True`` or the recipe's ``config_hash`` (integrity: what
    was approved is what runs). Returns a refusal-with-estimate until confirmed.
    ``checkpoint_path`` enables partial-run recovery (resume completed source
    partitions on a rerun) when the pipeline's stages are resumability-safe.
    ``bootstrap_ray`` opts into auto-starting a local Ray head when none is reachable.
    ``calibration`` is optional because a prior ``smoke`` of this exact recipe already
    stored its measurements: they are applied when none is passed, and the resource plan
    records that it planned from them.

    On success every step that persisted output is published as a content-addressed
    artifact, so a later request can reuse it instead of recomputing it
    (``REUSE_ARCHITECTURE.md``). ``goal`` records what the run was FOR (it is what makes a
    reuse candidate legible to a human later), and ``reuse`` carries the lineage when this
    run is itself the tail of a materialized continuation. Continuation identity
    is accepted only through an internal context: the core verifies one complete
    artifact against the original logical recipe and current source, then derives
    the publication tuples and complete run-record chain itself.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. A mismatch is refused before execution.
    """
    rec = _as_recipe(recipe).freeze()
    if goal is not None and not isinstance(goal, dict):
        return {
            "status": "refused",
            "reason": (
                "goal must be a JSON object/mapping, "
                f"got {type(goal).__name__}"
            ),
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
        }
    pviol = _safety.path_violations(
        [data, output_dir, checkpoint_path, *_safety.recipe_path_params(rec)]
    )
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "recipe_id": rec.recipe_id,
            "violations": pviol,
        }
    binding = _dataset_binding(rec, data)
    dp_obj = _profile_binding(binding)
    data_profile = dp_obj.to_dict() if dp_obj else None
    if dp_obj is not None and _profile_error(dp_obj):
        return _safety.redact(_profile_refusal(binding, dp_obj))

    # Only an explicit ``True`` or a hash string counts as confirmation. Gating on
    # ``confirm is False`` alone let any other falsy value (None / 0 / [] / {}) -- e.g. a
    # JSON-RPC ``confirm: null`` forwarded by the MCP adapter -- slip past BOTH this refusal
    # and the config_hash integrity check below into a silent full-scale run.
    if confirm is not True and not isinstance(confirm, str):
        return {
            "status": "refused",
            "reason": "full run requires explicit confirmation (0 silent full-scale runs)",
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
            "estimate": _estimate(data_profile),
            "confirm_with": f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed",
            "data_binding": binding.to_dict(),
        }
    if isinstance(confirm, str) and confirm != rec.config_hash:
        return {
            "status": "refused",
            "reason": "plan-execution integrity check failed: confirmed hash does not match the recipe",
            "confirmed": confirm,
            "config_hash": rec.config_hash,
            "data_binding": binding.to_dict(),
        }

    if _safety.require_smoke() and not _safety.verify_smoke_token(smoke_token, rec.config_hash):
        return {
            "status": "refused",
            "reason": "run requires smoke evidence (AUDIO_AGENT_REQUIRE_SMOKE is set): run smoke on this recipe and pass its 'smoke_token'",
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
            "data_binding": binding.to_dict(),
        }
    if _binding_blocks_execution(binding, data):
        return _safety.redact(_binding_refusal(binding))

    logical_identity, lineage_error = _verify_continuation_context(
        rec,
        binding,
        _continuation_context,
    )
    if lineage_error:
        return {
            "status": "refused",
            "reason": lineage_error,
            "data_binding": binding.to_dict(),
        }
    data_fp = None if logical_identity else (dp_obj.fingerprint() if dp_obj else None)
    dataset_key = (
        str(logical_identity["dataset_key"])
        if logical_identity
        else (dp_obj.dataset_key() if dp_obj else "")
    )
    fingerprint_tier = (
        str(logical_identity["fingerprint_tier"])
        if logical_identity
        else (dp_obj.fingerprint_tier if dp_obj else "")
    )
    recorded_data = (
        logical_identity.get("data_source")
        if logical_identity
        else (binding.primary_path or data)
    )

    stages, issues = build_stages(rec)
    if stages is None:
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        build_error = "; ".join(
            str(issue.get("message") or issue.get("code") or "")
            for issue in issues
        )
        return _safety.redact(
            {
                "status": "error",
                "recipe_id": rec.recipe_id,
                "issues": issues,
                "diagnosis": diagnose_failure(
                    build_error,
                    operation="run",
                    phase="stage_construction",
                ),
                "data_binding": binding.to_dict(),
            }
        )
    pretrain_finalizer, pretrain_error = _pretrain_finalizer(stages)
    if pretrain_error:
        return _safety.redact(
            {
                "status": "refused",
                "reason": pretrain_error,
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )

    caller_executor = executor
    execution_target = (
        "custom_executor"
        if caller_executor is not None
        else _ray_execution_target(os.environ.get("RAY_ADDRESS"))
    )
    preflight_env = probe_env()
    from nemo_curator.audio_agent.diagnostics import environment_preflight

    environment_decision = environment_preflight(
        stages,
        preflight_env,
        operation="run",
        execution_target=execution_target,
    )
    if not environment_decision.get("can_execute", False):
        return _safety.redact(
            {
                "status": "refused",
                "reason_code": "environment_action_required",
                "reason": environment_decision.get("summary"),
                "environment_decision": environment_decision,
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    owned_ray_address: str | None = None
    ray_address = (
        os.environ.get("RAY_ADDRESS")
        if caller_executor is None
        else None
    )
    if bootstrap_ray and caller_executor is None:
        try:
            from nemo_curator.audio_agent._ray import owns_cluster

            owned_before = owns_cluster()
            ray_address = _bootstrap_ray()
            if not owned_before and owns_cluster(ray_address):
                owned_ray_address = ray_address
        except Exception as exc:  # noqa: BLE001 - structured preflight failure
            from nemo_curator.audio_agent.diagnostics import diagnose_failure

            error_text = f"{type(exc).__name__}: {exc}"
            return _safety.redact(
                {
                    "status": "error",
                    "reason": (
                        "Ray bootstrap failed before resource planning: "
                        + error_text
                    ),
                    "diagnosis": diagnose_failure(
                        error_text,
                        stages=stages,
                        env=preflight_env,
                        operation="run",
                        phase="ray_bootstrap",
                        execution_target=execution_target,
                    ),
                    "recipe_id": rec.recipe_id,
                    "config_hash": rec.config_hash,
                    "data_binding": binding.to_dict(),
                }
            )
    try:
        env_obj = _resource_environment(
            preflight_env,
            ray_address,
            execution_target,
        )
        env = env_obj.to_dict()
        calibration, calibration_note = _calibration_for_run(calibration, rec.config_hash)
        rplan = _plan_resources(
            stages,
            env_obj,
            data_profile,
            calibration=calibration,
        )
        if calibration_note:
            rplan.notes.append(calibration_note)
        _adapt_resource_plan_for_target(
            rplan,
            execution_target=execution_target,
            operation="run",
        )
    except Exception as exc:  # noqa: BLE001 - public verb returns structured failures
        ray_cleanup = _shutdown_owned_ray(owned_ray_address)
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        error_text = f"{type(exc).__name__}: {exc}"
        return _safety.redact(
            {
                "status": "error",
                "reason": (
                    "resource planning failed: "
                    + error_text
                ),
                "diagnosis": diagnose_failure(
                    error_text,
                    stages=stages,
                    env=preflight_env,
                    operation="run",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
                **(
                    {"ray_bootstrap_cleanup": ray_cleanup}
                    if ray_cleanup is not None
                    else {}
                ),
            }
        )
    rec.with_machine_plan(rplan.to_dict(), machine_fingerprint=rplan.machine_fingerprint)
    if not rplan.feasible:
        ray_cleanup = _shutdown_owned_ray(owned_ray_address)
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        infeasible = "; ".join(rplan.escalations) or "resource plan infeasible"
        return _safety.redact(
            {
                "status": "refused",
                "reason": "resource plan is infeasible on the selected execution target",
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "escalations": rplan.escalations,
                "machine_plan": rplan.to_dict(),
                "diagnosis": diagnose_failure(
                    infeasible,
                    stages=stages,
                    env=env_obj,
                    operation="run",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "data_binding": binding.to_dict(),
                **(
                    {"ray_bootstrap_cleanup": ray_cleanup}
                    if ray_cleanup is not None
                    else {}
                ),
            }
        )

    failures: list[dict[str, Any]] = []
    runtime_diagnosis: dict[str, Any] | None = None
    results: list[Any] | None = None
    used_mode: str | None = None
    pretrain_prepared = False
    pretrain_output_rows: int | None = None
    started_at = _utc_now()
    t0 = time.perf_counter()
    try:
        if pretrain_finalizer is not None:
            pretrain_finalizer.prepare()
            pretrain_prepared = True
        results, used_mode = _run_pipeline_autofallback(
            stages,
            rplan.mode,
            caller_executor,
            checkpoint_path=checkpoint_path,
            before_retry=(
                pretrain_finalizer.prepare
                if pretrain_finalizer is not None
                else None
            ),
        )
        if pretrain_finalizer is not None:
            finalized_rows = pretrain_finalizer.finalize()
            if isinstance(finalized_rows, int) and not isinstance(finalized_rows, bool):
                pretrain_output_rows = max(0, finalized_rows)
    except Exception as e:  # noqa: BLE001 - classify + report, do not crash the caller
        if pretrain_finalizer is not None and pretrain_prepared:
            with contextlib.suppress(Exception):
                pretrain_finalizer.finalize(successful=False)
        from nemo_curator.audio_agent.diagnostics import diagnose_failure

        runtime_diagnosis = diagnose_failure(
            f"{type(e).__name__}: {e}",
            stages=stages,
            env=env_obj,
            operation="run",
            phase="pipeline_execution",
            execution_target=execution_target,
        )
        failures.append(dict(runtime_diagnosis.get("failure") or {}))
    elapsed = time.perf_counter() - t0
    ended_at = _utc_now()
    if used_mode is not None and used_mode != rplan.mode:
        actual_plan = rplan.to_dict()
        actual_plan["planned_mode"] = rplan.mode
        actual_plan["mode"] = used_mode
        actual_plan.setdefault("notes", []).append(
            f"executor fell back from {rplan.mode} to {used_mode}"
        )
        rec.with_machine_plan(
            actual_plan,
            machine_fingerprint=rplan.machine_fingerprint,
        )

    # Download-capable sources have no bytes to identify before their first
    # successful run. Re-resolve read-only after execution so that first run can
    # still publish artifacts under the dataset that was actually materialized.
    if not failures and dp_obj is None and binding.generated:
        refreshed = _dataset_binding(rec, data)
        refreshed_profile = _profile_binding(refreshed)
        if refreshed_profile is not None:
            binding = refreshed
            dp_obj = refreshed_profile
            data_profile = dp_obj.to_dict()
            if logical_identity is None:
                data_fp = dp_obj.fingerprint()
                dataset_key = dp_obj.dataset_key()
                fingerprint_tier = dp_obj.fingerprint_tier
                recorded_data = binding.primary_path or data

    output_paths = _recipe_outputs(rec, output_dir)
    report_obj = build_run_report(
        recipe=rec,
        result_tasks=results,
        data_profile=data_profile,
        env_profile=env,
        output_paths=output_paths,
        elapsed_sec=elapsed,
        failures=failures,
        examples=_examples(results, limit=5) if results else [],
        next_action="review retained/rejected; adjust thresholds and re-run if needed"
        if not failures
        else "triage the failure_reasons and re-validate",
    )
    if pretrain_output_rows is not None:
        report_obj.accepted = pretrain_output_rows
    terminal_outputs, cardinality_proven = _terminal_evidence_outputs(
        rec,
        list(getattr(report_obj, "output_paths", []) or []),
    )
    # Read the output back on EVERY run, not only one that declared a success contract. The
    # counts below are what tell a caller its 4-row manifest has 3 blank rows, and gating them
    # on acceptance criteria left the runs nobody was checking as the only ones reporting
    # nothing. The scan streams and keeps counts, so it costs a pass over the manifest the run
    # just wrote -- and it is now that pass rather than a second one: _acceptance_result is
    # handed this result instead of repeating the read.
    _preview, output_scan = _scan_terminal_output(terminal_outputs, limit=0)
    report_obj.source_items = int(getattr(report_obj, "input_count", 0))
    report_obj.output_rows = int(getattr(report_obj, "accepted", 0))
    report_obj.output_rows_written = rows_written_in(output_scan)
    report_obj.sparse_fields = sparse_fields_in(output_scan)
    report_obj.cardinality_proven = cardinality_proven
    report_obj.rejected = (
        max(0, report_obj.source_items - report_obj.output_rows)
        if cardinality_proven
        else None
    )

    from nemo_curator.audio_agent import run_store

    run_id = run_store.new_run_id(rec.config_hash)
    # Declared before the derivation below so that one can report its own failure through the
    # same channel the publication step already uses, rather than returning empty in silence.
    provenance: dict[str, Any] = {
        "run_record_persisted": False,
        "artifacts_published": 0,
        "warnings": [],
    }
    persistence_warnings: list[str] = provenance["warnings"]
    roles, keys = _produced_roles_keys(stages, data_profile, warnings=persistence_warnings)
    acceptance_result = _acceptance_result(
        rec,
        report_obj,
        roles,
        keys,
        list(getattr(report_obj, "output_paths", []) or []),
        output_scan=output_scan,
    )
    published: list[dict[str, Any]] = []
    if not failures:
        # Only a run that completed may publish: a crashed run's partial output must never
        # acquire the _COMPLETE marker that makes it look reusable.
        published = _publish_artifacts(
            rec,
            stages,
            dataset_key=dataset_key,
            fingerprint_tier=fingerprint_tier,
            per_stage=getattr(report_obj, "per_stage_metrics", {}) or {},
            run_id=run_id,
            input_count=int(getattr(report_obj, "input_count", 0)),
            data_profile=data_profile,
            # A continuation's profile describes the artifact it physically read, not the corpus
            # its logical dataset key names, so recording that as the corpus coverage would
            # describe one dataset with another's file list. Withholding it costs a later delta
            # (which says so by name) instead of computing one against the wrong inventory.
            inventory=(None if logical_identity else _consumed_inventory(rec, dp_obj)),
            started_at=started_at,
            ended_at=ended_at,
            elapsed_sec=elapsed,
            step_identity=(
                list(logical_identity["step_identity"])
                if logical_identity
                else None
            ),
            persistence_warnings=persistence_warnings,
        )
    provenance["artifacts_published"] = len(published)
    lineage = {**(reuse or {}), "published": published}
    _record_run(
        rec,
        run_id=run_id,
        data=recorded_data,
        data_fp=data_fp,
        dataset_key=dataset_key,
        fingerprint_tier=fingerprint_tier,
        report=report_obj,
        failed=bool(failures),
        goal=goal,
        elapsed=elapsed,
        env=env,
        produced_roles=roles,
        produced_keys=keys,
        acceptance_result=acceptance_result,
        reuse=lineage,
        logical_steps=(
            list(logical_identity["logical_steps"])
            if logical_identity
            else None
        ),
        persistence_status=provenance,
    )
    result = {
        "status": "completed" if not failures else "failed",
        "run_id": run_id,
        "report": report_obj.to_dict(),
        "acceptance": acceptance_result,
        "machine_plan": rec.machine_plan,
        "environment_decision": environment_decision,
        "reuse": lineage,
        "provenance": provenance,
        "data_binding": {
            **binding.to_dict(),
            **(
                {
                    "logical_dataset_key": dataset_key,
                    "logical_data_source": recorded_data,
                }
                if logical_identity
                else {}
            ),
        },
    }
    if runtime_diagnosis is not None:
        result["diagnosis"] = runtime_diagnosis
    warnings = list(persistence_warnings)
    if output_dir:
        warnings.append(
            "output_dir is a legacy no-op and was not reported as an output; "
            "configure output paths on recipe stages."
        )
    if warnings:
        result["warnings"] = warnings
    ray_cleanup = _shutdown_owned_ray(owned_ray_address)
    if ray_cleanup is not None:
        result["ray_bootstrap_cleanup"] = ray_cleanup
        if not ray_cleanup:
            result.setdefault("warnings", []).append(
                "the locally bootstrapped Ray head could not be stopped safely"
            )
    return _safety.redact(result)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _produced_roles_keys(
    stages: list[Any],
    data_profile: dict[str, Any] | None,
    *,
    warnings: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Cumulative semantic roles + literal keys the built pipeline produces.

    Same derivation the resume-safety guard uses, so an artifact advertises exactly the roles
    a later suffix will be re-validated against.

    A failure still returns ``([], [])`` -- provenance detail is never worth failing a completed
    run over -- but it now SAYS so through ``warnings``. Empty and unknown are indistinguishable
    downstream: ``acceptance._completeness`` reads an empty declared set as "no producer
    evidence" and reports the vaguer of two true-ish answers, so a catalogue-wide contract
    failure would quietly cost every acceptance message its precision with nothing anywhere
    recording that the derivation broke.

    Note this is NOT the artifact's ``produced_roles``: ``_publish_artifacts`` derives those
    itself, seeded from ``_derive_initial`` and suppressing failures per stage, so one
    unreadable contract cannot empty them. What this feeds is the run record and the acceptance
    evidence.
    """
    try:
        from nemo_curator.stages.audio import agent as foundation
        from nemo_curator.stages.audio._conformance import produced_roles

        roles, keys = _derive_initial(data_profile)
        for st in stages:
            c = foundation.build_contract(st)
            roles |= produced_roles(c)
            keys |= set(c.writes.data_keys) | set(c.writes.segment_data_keys)
        roles.discard("unknown")
        return sorted(roles), sorted(keys)
    except Exception as exc:  # noqa: BLE001 - provenance detail, never worth failing a completed run
        if warnings is not None:
            warnings.append(
                "produced roles/keys could not be derived, so the run record and the acceptance "
                f"evidence report none rather than unknown: {type(exc).__name__}: {exc}"
            )
        return [], []


def _stage_cost(per_stage: dict[str, Any], stage: Any) -> tuple[float, float]:  # noqa: ANN401
    """``(duration_sec, gpu_seconds)`` measured for one stage, 0 when unmeasured.

    Real numbers only -- an unmeasured stage reports 0 rather than a share of the total, so a
    "time saved" estimate is never inflated by guesswork.
    """
    duration = stage_duration_sec(per_stage, str(getattr(stage, "name", "") or ""))
    gpus = float(getattr(getattr(stage, "resources", None), "gpus", 0) or 0)
    return round(duration, 3), round(duration * gpus, 3)


def _consumed_inventory(rec: Recipe, dp: Any) -> dict[str, str] | None:  # noqa: ANN401 - DataProfile
    """The files this run actually read, which is what its artifacts cover.

    Normally every file the profiler found. A run whose source was narrowed to a named subset
    (``include_files``, how a delta processes only what changed) consumed only those, and
    recording the whole corpus as its coverage would claim results it never computed.
    """
    if dp is None or not dp.inventory:
        return None
    named = rec.stages[0].params.get("include_files") if rec.stages else None
    if not isinstance(named, list):
        return dict(dp.inventory)
    root = dp.inventory_root or ""
    wanted = {os.path.relpath(os.path.abspath(os.path.expanduser(str(p))), root) for p in named}
    return {rel: token for rel, token in dp.inventory.items() if rel in wanted}


def _publish_artifacts(  # noqa: PLR0913 - publishing gathers the run's whole context
    rec: Recipe,
    stages: list[Any],
    *,
    dataset_key: str,
    fingerprint_tier: str,
    per_stage: dict[str, Any],
    run_id: str,
    input_count: int,
    data_profile: dict[str, Any] | None,
    inventory: dict[str, str] | None = None,
    started_at: str,
    ended_at: str,
    elapsed_sec: float = 0.0,
    step_identity: list[tuple[str, str, int]] | None = None,
    persistence_warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Register every step that actually persisted output as a reusable artifact.

    A step with no output-location param writes nothing, so it can never be a resume point and
    gets no artifact -- reuse always resumes from disk. ``step_identity`` overrides the
    ``(step_key, input_key, stage_index)`` of each step positionally, which is how a
    materialized continuation registers its tail under the identity of the pipeline the user
    actually asked for. Best-effort throughout: a bookkeeping problem must not turn a
    successful run into a failure.
    """
    # Unknown input is not a reusable namespace.  Publishing under ``""`` would
    # let a later unknown input look identical merely because neither was
    # profiled.
    if not dataset_key:
        if persistence_warnings is not None:
            persistence_warnings.append(
                "artifacts were not published because source data identity is unavailable"
            )
        return []

    try:
        from nemo_curator.audio_agent import artifacts as art_mod
        from nemo_curator.stages.audio import agent as foundation
        from nemo_curator.stages.audio._conformance import produced_roles as _roles_of
    except Exception as exc:  # noqa: BLE001
        if persistence_warnings is not None:
            persistence_warnings.append(
                "artifact registry could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            )
        return []

    out: list[dict[str, Any]] = []
    try:
        plans = art_mod.plan_steps(rec, dataset_key)
    except Exception as exc:  # noqa: BLE001
        if persistence_warnings is not None:
            persistence_warnings.append(
                "artifact publication plan could not be built: "
                f"{type(exc).__name__}: {exc}"
            )
        return []

    roles, keys = _derive_initial(data_profile)
    rows_in = input_count
    cumulative = 0.0
    for i, (plan, st) in enumerate(zip(plans, stages, strict=False)):
        with contextlib.suppress(Exception):
            c = foundation.build_contract(st)
            roles |= _roles_of(c)
            keys |= set(c.writes.data_keys) | set(c.writes.segment_data_keys)
        duration, gpu_seconds = _stage_cost(per_stage, st)
        cumulative += duration  # every step so far, persisting or not: that is the true saving
        if not plan.persists():
            continue
        if "://" in plan.uri and not plan.uri.startswith("file://"):
            if persistence_warnings is not None:
                persistence_warnings.append(
                    "artifact publication/reuse is unsupported for non-local "
                    f"output backend {plan.uri!r}"
                )
            continue
        if not os.path.exists(os.path.expanduser(plan.uri)):
            if persistence_warnings is not None:
                persistence_warnings.append(
                    f"declared persisted output {plan.uri!r} was not found; "
                    f"{plan.stage_ref} was not published for reuse"
                )
            continue
        # Reusing the LAST step skips the run outright, so its true cost is the wall clock,
        # not the sum of stage timers (which misses setup, scheduling and teardown).
        through_here = max(cumulative, elapsed_sec) if plan.index == len(plans) - 1 else cumulative
        step_key, input_key, stage_index = (
            step_identity[i] if step_identity and i < len(step_identity)
            else (plan.step_key, plan.input_key, plan.index)
        )
        art = art_mod.Artifact(
            step_key=step_key,
            input_key=input_key,
            stage_ref=plan.stage_ref,
            stage_index=stage_index,
            semantic_params=plan.semantic_params,
            contract_hash=rec.contract_hash,
            uri=plan.uri,
            # Re-classified now rather than trusting the plan-time guess: at plan time a directory
            # that does not exist yet cannot be identified, and the kind decides which source stage
            # may re-read this artifact later.
            kind=art_mod.classify_output(plan.uri) or plan.kind,
            rows_in=rows_in,
            produced_roles=sorted(roles - {"unknown"}),
            produced_keys=sorted(keys),
            duration_sec=duration,
            cumulative_sec=round(through_here, 3),
            gpu_seconds=gpu_seconds,
            device="gpu" if gpu_seconds else "cpu",
            started_at=started_at,
            ended_at=ended_at,
            dataset_key=dataset_key,
            fingerprint_tier=fingerprint_tier,
            covers_files=len(inventory or {}),
            impl_version=plan.impl_version,
            code_version=art_mod.code_version(),
            model_version=plan.model_version,
            deterministic=plan.deterministic,
            ttl_sec=plan.ttl_sec,
            run_id=run_id,
        )
        try:
            art_mod.publish(art)
            if inventory:
                # After publish, so a coverage file never outlives the artifact it describes.
                art_mod.save_coverage(art.step_key, inventory)
        except Exception as exc:  # noqa: BLE001 - compute completion stays separate
            if persistence_warnings is not None:
                persistence_warnings.append(
                    f"artifact publication failed for {plan.uri!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue
        rows_in = art.rows_out or rows_in
        out.append({"step_key": art.step_key, "stage": art.stage_ref, "uri": art.uri, "rows": art.rows_out})
    return out


def _record_run(  # noqa: PLR0913 - a provenance record intentionally gathers many fields
    rec: Recipe,
    *,
    run_id: str,
    data: str | None,
    data_fp: str | None,
    dataset_key: str,
    fingerprint_tier: str,
    report: Any,  # noqa: ANN401
    failed: bool,
    goal: dict[str, Any] | None = None,
    elapsed: float = 0.0,
    env: dict[str, Any] | None = None,
    produced_roles: list[str] | None = None,
    produced_keys: list[str] | None = None,
    acceptance_result: dict[str, Any] | None = None,
    reuse: dict[str, Any] | None = None,
    logical_steps: list[str] | None = None,
    persistence_status: dict[str, Any] | None = None,
) -> str:
    """Persist a local RunRecord (provenance for tracing + continuation). Best-effort."""
    from nemo_curator.audio_agent import run_store
    from nemo_curator.audio_agent.contracts import RunRecord

    env = env or {}
    record = RunRecord(
        run_id=run_id,
        # Redact secret-valued params (e.g. hf_token) so they never land in the on-disk record.
        recipe=_safety.redact(rec.to_dict(), redact_transcripts=False),
        config_hash=rec.config_hash,
        semantic_hash=rec.semantic_hash,
        contract_hash=rec.contract_hash,
        parent_run_id=rec.parent_run_id,
        goal=dict(goal or {}),
        data_source=data,
        data_fingerprint=data_fp,
        dataset_key=dataset_key,
        fingerprint_tier=fingerprint_tier,
        acceptance_criteria=list(rec.acceptance_criteria),
        acceptance_result=(
            acceptance_result
            if acceptance_result is not None
            else _acceptance_result(
                rec,
                report,
                produced_roles,
                produced_keys,
                list(getattr(report, "output_paths", []) or []),
            )
        ),
        status="failed" if failed else "completed",
        accepted=int(getattr(report, "accepted", 0)),
        input_count=int(getattr(report, "input_count", 0)),
        output_paths=list(getattr(report, "output_paths", []) or []),
        elapsed_sec=round(elapsed, 3),
        per_stage_metrics=dict(getattr(report, "per_stage_metrics", {}) or {}),
        env_summary={k: env.get(k) for k in ("has_gpu", "gpu_count", "gpu_names", "python_version", "cuda_runtime_version")},
        curator_version=str(env.get("curator_version") or ""),
        knowledge_version=rec.knowledge_version,
        # The chain of the pipeline the USER asked for. After a continuation ``rec`` is the
        # rewritten recipe, whose own keys describe a pipeline nobody requested and share no
        # prefix with the request -- which made every continued run unrecognisable to anything
        # matching on this field, including the disclosure of already-done-but-unsaved work.
        steps=logical_steps or _step_keys(rec, dataset_key),
        reuse=dict(reuse or {}),
        created_at=_utc_now(),
    )
    try:
        path = run_store.save(record)
    except Exception as exc:  # noqa: BLE001 - compute may succeed even if provenance does not
        if persistence_status is not None:
            persistence_status["run_record_persisted"] = False
            persistence_status.setdefault("warnings", []).append(
                "run record persistence failed: "
                f"{type(exc).__name__}: {exc}"
            )
        return record.run_id
    if persistence_status is not None:
        persistence_status["run_record_persisted"] = True
        persistence_status["run_record_path"] = path
    return record.run_id


def _step_keys(rec: Recipe, dataset_key: str) -> list[str]:
    try:
        from nemo_curator.audio_agent import artifacts as art_mod

        return art_mod.step_keys(rec, dataset_key)
    except Exception:  # noqa: BLE001
        return []


def _acceptance_result(
    rec: Recipe,
    report: Any,  # noqa: ANN401
    produced_roles: list[str] | None,
    produced_keys: list[str] | None,
    outputs: list[str] | None = None,
    *,
    output_scan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the run's own success contract against the evidence it just produced.

    Recording the OUTCOME (not just the criteria) is what lets a reuse candidate be judged
    later without re-deriving whether it actually met its bar. A contract that could not be
    verified is reported as ``unverifiable`` with the reason: swallowing it would let a run
    claim success while its success bar was never checked.

    ``outputs`` are read back as row-level evidence so the contract judges the DATA, not the
    role labels ``validate`` declared. Declaration-level checking passes a field that every
    row left null -- the precise failure the contract exists to prevent.

    ``output_scan`` lets a caller that already read the output hand the result over instead of
    paying for a second pass. It is accepted only if it names the same terminal output this
    contract would have scanned; anything else is re-read, so passing the wrong scan costs
    time rather than judging one run's contract against another run's bytes.
    """
    if not rec.acceptance_criteria:
        return {}
    try:
        evidence_outputs, cardinality_proven = _terminal_evidence_outputs(
            rec,
            list(outputs or []),
        )
        per_item: list[dict[str, Any]] = []
        if not _needs_terminal_evidence(rec):
            output_scan = {}
        elif not _scan_covers(output_scan, evidence_outputs):
            per_item, output_scan = _scan_terminal_output(evidence_outputs, limit=0)
        expected_output_rows = (
            int(getattr(report, "accepted", 0))
            if cardinality_proven
            else None
        )
        return verify(
            list(rec.acceptance_criteria),
            {
                "produced_roles": list(produced_roles or []),
                "produced_keys": list(produced_keys or []),
                "per_item": per_item,
                "output_scan": output_scan,
                "metrics": _aggregate_metrics(output_scan),
                "expected_output_rows": expected_output_rows,
                "retained": int(getattr(report, "accepted", 0)),
                "input_count": int(getattr(report, "input_count", 0)),
            },
        )
    except Exception as e:  # noqa: BLE001 - report the failure, but do not fail a completed run
        return {"overall": "unverifiable", "reason": f"could not verify the success contract: {e}"}


def _scan_covers(output_scan: dict[str, Any] | None, evidence_outputs: list[str]) -> bool:
    """Whether an already-taken scan read the output this contract has to judge.

    The scan records the location it read, so this is a comparison rather than a promise the
    caller has to keep. Sharing a read is only safe if it is provably the same read.
    """
    if not output_scan:
        return False
    terminal = str(evidence_outputs[-1]) if evidence_outputs else ""
    return str(output_scan.get("terminal_output") or "") == terminal


def _needs_terminal_evidence(rec: Recipe) -> bool:
    """Whether this contract needs an exhaustive manifest readback."""
    from nemo_curator.audio_agent.acceptance import parse_criteria

    return any(
        criterion.type == "output_completeness"
        or criterion.check.get("scope") == "per_retained_item"
        or (
            criterion.type in {"quality_standard", "distribution"}
            and (criterion.check.get("scope") or "aggregate") == "aggregate"
        )
        for criterion in parse_criteria(rec.acceptance_criteria)
    )


def _terminal_evidence_outputs(
    rec: Recipe,
    reported_outputs: list[str],
) -> tuple[list[str], bool]:
    """Select the actual per-item serializer and whether it is provably 1:1.

    ``output_dir`` is a legacy verb argument and is currently not injected into
    the recipe. It may therefore appear in ``reported_outputs`` despite not being
    written by the pipeline. Prefer a recipe-declared serializer so acceptance
    never scans that phantom path.

    Final ``ManifestWriterStage`` and ``DocumentBatchJsonlWriterStage`` sinks
    have proven one-row-per-logical-returned-row contracts.
    ``SnippetManifestWriterStage`` intentionally skips origin stubs, and a
    writer followed by another stage need not have the final run's cardinality,
    so those cases remain explicitly unproven.
    """
    per_item_writers = {
        "DocumentBatchJsonlWriterStage",
        "ManifestWriterStage",
        "SnippetManifestWriterStage",
    }
    for index in range(len(rec.stages) - 1, -1, -1):
        stage = rec.stages[index]
        short_ref = stage.ref.rsplit(".", 1)[-1]
        if short_ref not in per_item_writers:
            continue
        output = stage.params.get("output_path")
        if not isinstance(output, str) or not output:
            continue
        cardinality_proven = (
            short_ref
            in {"DocumentBatchJsonlWriterStage", "ManifestWriterStage"}
            and index == len(rec.stages) - 1
        )
        return [output], cardinality_proven

    declared_outputs = _recipe_outputs(rec, None)
    if declared_outputs:
        return [declared_outputs[-1]], False
    return ([str(reported_outputs[-1])] if reported_outputs else []), False


def _same_output_target(left: str, right: str) -> bool:
    """Canonical equality for local paths and lexical equality for remote URIs."""
    from nemo_curator.audio_agent.input_identity import canonical_source

    try:
        return canonical_source(left).rstrip("/") == canonical_source(right).rstrip("/")
    except (TypeError, ValueError):
        return False


def report(output: str, *, recipe: Recipe | dict[str, Any] | None = None, data: str | None = None) -> dict[str, Any]:
    """Post-hoc report from an output manifest/dir (counts rows vs input scale).

    Without ``recipe``, ``data`` is profiled directly for the input count. With
    a recipe, it is only an optional assertion matching the configured source.
    A supplied recipe also binds ``output`` to its declared terminal serializer
    and carries its frozen identity/acceptance contract into the report.
    """
    rec = _as_recipe(recipe).freeze() if recipe is not None else None
    pviol = _safety.path_violations(
        [output, data, *(_safety.recipe_path_params(rec) if rec is not None else [])]
    )
    if pviol:
        return {"status": "refused", "reason": "path(s) resolve outside the allowed workspace", "violations": pviol}
    binding = _dataset_binding(rec, data) if rec is not None else None
    if binding is not None and _binding_blocks_execution(binding, data):
        return _safety.redact(_binding_refusal(binding))
    dp_obj = _profile_binding(binding) if binding is not None else (profile_data(data) if data else None)
    if dp_obj is not None and _profile_error(dp_obj):
        if binding is not None:
            return _safety.redact(_profile_refusal(binding, dp_obj))
        return _safety.redact(
            {
                "status": "refused",
                "reason": f"input data is unreadable or malformed: {_profile_error(dp_obj)}",
                "data_profile": dp_obj.to_dict(),
            }
        )
    data_profile = dp_obj.to_dict() if dp_obj else None
    terminal_outputs: list[str] = []
    cardinality_proven = False
    if rec is not None:
        terminal_outputs, cardinality_proven = _terminal_evidence_outputs(rec, [])
        if not terminal_outputs:
            return _safety.redact(
                {
                    "status": "refused",
                    "reason": (
                        "the recipe declares no terminal output that can be "
                        "bound to this post-hoc report"
                    ),
                    "recipe_id": rec.recipe_id,
                    "config_hash": rec.config_hash,
                    "data_binding": binding.to_dict() if binding is not None else None,
                }
            )
        if not any(_same_output_target(output, declared) for declared in terminal_outputs):
            return _safety.redact(
                {
                    "status": "refused",
                    "reason": (
                        "reported output does not match the recipe's declared "
                        "terminal output"
                    ),
                    "output": output,
                    "declared_terminal_outputs": terminal_outputs,
                    "recipe_id": rec.recipe_id,
                    "config_hash": rec.config_hash,
                    "data_binding": binding.to_dict() if binding is not None else None,
                }
            )

    preview, output_scan = _scan_terminal_output([output], limit=5)
    locate_status = str(output_scan.get("status") or "unavailable")
    if locate_status == "no_manifest":
        inventory = _scan_output_inventory(output)
        if inventory.get("status") in {"complete", "empty"}:
            source_items = (
                int((data_profile or {}).get("num_files", 0))
                if data_profile is not None
                else None
            )
            rpt = build_run_report(
                recipe=rec or Recipe(),
                result_tasks=[],
                data_profile=data_profile,
                env_profile=probe_env().to_dict(),
                output_paths=[output],
                examples=[],
                next_action=(
                    "review the output inventory; row-level fields are unavailable "
                    "for this output type"
                ),
            )
            d = rpt.to_dict()
            d.update(
                {
                    "status": "ok",
                    "accepted": None,
                    "rejected": None,
                    "input_count": source_items,
                    "source_items": source_items,
                    "output_rows": None,
                    "output_files": int(inventory.get("files") or 0),
                    "output_scan": output_scan,
                    "output_inventory": inventory,
                }
            )
            if rec is not None and rec.acceptance_criteria:
                d["acceptance"] = verify(
                    list(rec.acceptance_criteria),
                    {
                        "per_item": [],
                        "output_scan": output_scan,
                        "metrics": {},
                        "expected_output_rows": None,
                        "retained": None,
                        "input_count": source_items,
                    },
                    recipe=rec,
                )
            if binding is not None:
                d["data_binding"] = binding.to_dict()
            return _safety.redact(d)
    if locate_status in {"missing", "no_manifest", "unreadable", "unavailable"}:
        return _safety.redact(
            {
                "status": "error",
                "reason": f"terminal output could not be read as a manifest: {locate_status}",
                "output": output,
                "output_scan": output_scan,
                "recipe_id": rec.recipe_id if rec is not None else None,
                "config_hash": rec.config_hash if rec is not None else None,
                "data_binding": binding.to_dict() if binding is not None else None,
            }
        )

    accepted = int(output_scan.get("valid_rows") or 0)
    input_count = (
        int((data_profile or {}).get("num_files", 0))
        if data_profile is not None
        else None
    )
    corrupt = bool(
        int(output_scan.get("read_errors") or 0)
        or int(output_scan.get("malformed_rows") or 0)
        or int(output_scan.get("blank_rows") or 0)
    )
    failures = (
        [
            {
                "code": "terminal_output_incomplete",
                "message": (
                    "terminal output contains unreadable, malformed, or blank "
                    "rows; accepted counts include valid JSON-object rows only"
                ),
                "output_scan": output_scan,
            }
        ]
        if corrupt
        else []
    )
    rpt = build_run_report(
        recipe=rec or Recipe(),
        result_tasks=[],
        data_profile=data_profile,
        env_profile=probe_env().to_dict(),
        output_paths=[output],
        failures=failures,
        examples=preview,
        next_action="compare against expected retention; scale up or adjust thresholds",
    )
    d = rpt.to_dict()
    d["status"] = "error" if corrupt else "ok"
    d["accepted"] = accepted
    d["input_count"] = input_count
    d["source_items"] = input_count
    d["output_rows"] = accepted
    # Output-row cardinality is not generally the same as source-item
    # cardinality (split/fan-out stages are common).  Only a proven 1:1
    # terminal serializer supports subtraction as a rejection count.
    d["rejected"] = (
        max(0, input_count - accepted)
        if input_count is not None and cardinality_proven
        else None
    )
    d["output_scan"] = output_scan
    if rec is not None and rec.acceptance_criteria:
        d["acceptance"] = verify(
            list(rec.acceptance_criteria),
            {
                "per_item": [],
                "output_scan": output_scan,
                "metrics": _aggregate_metrics(output_scan),
                "expected_output_rows": accepted if cardinality_proven else None,
                "retained": accepted,
                # A relative-yield denominator must come from actual source
                # evidence; substituting accepted would manufacture 100%.
                "input_count": input_count,
            },
            recipe=rec,
        )
    if binding is not None:
        d["data_binding"] = binding.to_dict()
    return _safety.redact(d)


def verify(
    acceptance_criteria: list[dict[str, Any]],
    evidence: dict[str, Any] | None = None,
    *,
    frozen_criteria: list[dict[str, Any]] | None = None,
    recipe: Recipe | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate acceptance criteria against gathered evidence -> AcceptanceReport (1A.1/1A.3).

    Deterministic verifier: it runs nothing itself, it judges the ``evidence`` the
    host assembled (from ``validate`` — ``produced_roles``/``produced_keys`` — and
    from ``smoke``/``run`` — ``metrics``/``per_item``/``retained``/``input_count``,
    plus optional ``unachievable_fields``). Returns per-criterion states
    (met / not_met / unverifiable / unachievable) and an ``overall`` that is
    ``met`` iff every ``must`` criterion is met — the anti-goalpost-moving gate.

    Pass the confirmed contract as ``frozen_criteria`` (or a ``recipe`` carrying
    ``acceptance_criteria``) to run the honesty guard (1A.3): if the criteria being
    verified are weaker than confirmed, it is flagged and ``overall`` is ``not_met``.
    """
    from nemo_curator.audio_agent.acceptance import parse_criteria
    from nemo_curator.audio_agent.acceptance import verify as _verify

    frozen = None
    if recipe is not None:
        frozen = parse_criteria(_as_recipe(recipe).acceptance_criteria)
        if frozen_criteria is not None:
            explicit_frozen = parse_criteria(frozen_criteria)
            if [criterion.to_dict() for criterion in explicit_frozen] != [
                criterion.to_dict() for criterion in frozen
            ]:
                msg = (
                    "frozen_criteria conflicts with recipe.acceptance_criteria; "
                    "provide one honesty source or make them identical"
                )
                raise ValueError(msg)
    elif frozen_criteria is not None:
        frozen = parse_criteria(frozen_criteria)
    report_obj = _verify(
        parse_criteria(acceptance_criteria),
        evidence if evidence is not None else {},
        frozen_criteria=frozen,
    )
    return _safety.redact(report_obj.to_dict())


def runs(
    run_id: str | None = None,
    *,
    data: str | None = None,
    stage: str | None = None,
    since: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List local run records, or load one by ``run_id`` (provenance for tracing).

    Local history only — NOT shared memory / cross-user learning. Use it to trace a
    prior run or to feed ``plan_continuation`` for a follow-up request. ``data`` /
    ``stage`` / ``since`` query the index for what has already been done to a corpus.
    ``data`` takes either the dataset's path or a ``dataset_key`` copied from earlier
    reuse output — profiling a key as if it were a path would silently match nothing.
    """
    from nemo_curator.audio_agent import run_index, run_store

    if run_id:
        rec = run_store.load(run_id)
        return _safety.redact(rec.to_dict()) if rec else {"error": f"no run record {run_id!r}"}
    if data or stage or since:
        pviol = _safety.path_violations(
            [data]
            if data and not data.startswith(tuple(f"{tier}:" for tier in DATASET_KEY_TIERS))
            else []
        )
        if pviol:
            return {
                "status": "refused",
                "reason": "path(s) resolve outside the allowed workspace",
                "violations": pviol,
            }
        dataset_key = _dataset_key_arg(data) if data else None
        return _safety.redact(
            {
                "dataset_key": dataset_key,
                "runs": run_index.find_runs(dataset_key=dataset_key, since=since, limit=limit),
                "artifacts": run_index.find_artifacts(
                    dataset_key=dataset_key,
                    stage_ref=stage,
                    since=since,
                    limit=limit,
                ),
            }
        )
    # ``limit`` applies here too. The filtered branch above has always honoured it while this one
    # returned every record ever written, so the argument silently meant different things
    # depending on whether a filter happened to be supplied.
    return {"runs": run_store.list_runs()[: int(limit)] if int(limit) >= 0 else run_store.list_runs()}


def _dataset_key_arg(data: str) -> str:
    """A dataset identity from either a path or a key already printed by an earlier scan."""
    from nemo_curator.audio_agent.contracts import DATASET_KEY_TIERS

    if data.startswith(tuple(f"{t}:" for t in DATASET_KEY_TIERS)):
        return data
    return profile_data(data).dataset_key()


def reindex() -> dict[str, Any]:
    """Rebuild the run/artifact index from the JSON records, and drop the knowledge cache.

    The index is a cache; the JSON records are the source of truth. Run this after moving,
    pruning, or hand-editing ``.audio_agent_runs/``.

    The knowledge index is a second cache, held for the life of the process because the YAML is
    static (``index.get_index`` is ``lru_cache``d). That is right for a run, and a trap for the
    person editing a card: their change was invisible until a restart, from the one verb whose
    name promises otherwise. Clearing it here costs a re-read of the card directory on the next
    call and makes ``reindex`` mean what it says.
    """
    from nemo_curator.audio_agent import run_index
    from nemo_curator.audio_agent.index import get_index

    get_index.cache_clear()

    return run_index.reindex()


def reuse_scan(recipe: Recipe | dict[str, Any], *, data: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Find prior work this recipe could reuse, and describe it for a human decision.

    Probes the artifact registry with the recipe's step keys and returns the longest safely
    reusable prefix plus ranked approval cards (objective, pipeline, input/output, key params,
    date, metrics, estimated saving). Read-only: it never reuses anything by itself.

    ``prompt_user`` is the anti-nag signal — false when there is no candidate at all, or when
    the saving was measured and is too small to be worth a question (take it and disclose it).
    An unmeasured prefix containing work the cards call expensive is named in ``unpriced_stages``
    and always asks: not knowing what something cost is not the same as it having been cheap.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. Omit it when the recipe is already unambiguous.
    """
    from nemo_curator.audio_agent import reuse as _reuse

    rec = _as_recipe(recipe).freeze()
    pviol = _safety.path_violations([data, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    binding = _dataset_binding(rec, data)
    dp = _profile_binding(binding)
    profile_error = _profile_error(dp) if dp is not None else ""
    dataset_key = (
        dp.dataset_key()
        if dp and not profile_error and not _binding_blocks_execution(binding, data)
        else ""
    )
    result = _reuse.scan(rec, dataset_key=dataset_key, limit=limit)
    result["data_binding"] = binding.to_dict()
    if profile_error:
        result["rationale"] = f"prior work was not considered: {profile_error}"
        result["data_profile"] = dp.to_dict()
    if _binding_blocks_execution(binding, data):
        result["rationale"] = f"prior work was not considered: {binding.reason}"
    if result.get("decision") == "fresh" and dataset_key and dp is not None:
        _attach_delta(result, rec, dp)
    return _safety.redact(result)


def _attach_delta(result: dict[str, Any], rec: Recipe, dp: Any) -> None:  # noqa: ANN401 - DataProfile
    """Add the changed-file option to a card whose key missed, when there is one.

    A miss is where this belongs: a key names the whole corpus, so adding one file misses every
    key even though almost all of the work behind it still holds. Left alone the card says
    "nothing matches" and the honest reading of that is "recompute everything" -- which is how a
    one-file addition comes to cost a full run.

    A refusal is attached too when the change itself was understood. "Prior work exists but
    these stages cannot be split per file" is a different fact from "this has never run", it is
    the reason the full run is unavoidable, and it names what would have to change for it not
    to be -- usually a checkpoint, or a stage that has not declared per-row independence.
    """
    from nemo_curator.audio_agent import delta as _delta

    decision = _delta.plan(
        rec,
        dataset_key=dp.dataset_key(),
        inventory=dp.inventory or None,
        inventory_root=dp.inventory_root,
    )
    if decision.status != "ready" and decision.change is None and not result.get("prior_on_other_data"):
        # Nothing ran before, so there is nothing a delta could have narrowed and no reason to
        # mention one. When a prior run IS known, the same silence would hide why this costs
        # full price -- usually an inventory that was never recorded, which is fixable.
        return
    result["delta"] = decision.to_dict()
    if decision.status != "ready":
        result["rationale"] = (
            f"{result.get('rationale', '')}; running only the changed file(s) is not available "
            f"here: {decision.reason}"
        ).lstrip("; ")
        return
    change = decision.change
    result.update(
        # The key really did miss. But "fresh" is the one word a host reads as "recompute
        # everything", and one did exactly that on a single-file addition -- recurating three
        # files and reporting a checkpoint was missing -- while `recommended: delta` sat two
        # fields away unread. `decision` names the cheapest correct action; the miss it rests
        # on stays visible in `key_matched` and in the rationale below, so nothing is hidden.
        decision="delta",
        key_matched=False,
        recommended="delta",
        prompt_user=True,
        estimated_saving_sec=decision.estimated_saving_sec,
        saving_is_lower_bound=True,
        choices=[
            {
                "id": "delta",
                "label": f"Run the {len(decision.files)} changed file(s) only",
                "effect": (
                    f"run {decision.prefix} stage(s) over the changed file(s) and merge the rows into "
                    f"the prior result, keeping {decision.keeps} row(s) and replacing {decision.drops}"
                ),
                "verb": "delta_run",
            },
            {"id": "fresh", "label": "Run fresh", "effect": "recompute every file from the start"},
        ],
        rationale=(
            f"no artifact matches this exact corpus, but {change.phrase()} since a prior run of this same "
            f"pipeline: the other {len(change.unchanged)} file(s)' results through {decision.stage_ref} still "
            f"hold, so delta_run processes only the changed file(s) and merges them in"
        ),
    )


def add_checkpoint(
    recipe: Recipe | dict[str, Any],
    *,
    output_path: str | None = None,
    after: str | None = None,
) -> dict[str, Any]:
    """Where a mid-pipeline manifest would make the expensive work reusable, and the recipe for it.

    Without ``output_path`` this answers only where such a checkpoint may go and what a later
    run would then skip. With one, it returns the same recipe carrying a ``ManifestWriterStage``
    at that position -- as a recipe to look at, never written to disk and never run. Adding it
    is a recipe change the user makes, so it goes back through ``validate`` -> confirm ->
    ``run`` like any other, and the checkpoint is a file they chose rather than a hidden cache.

    Both halves of the position are simulated rather than assumed: a manifest cannot serialize a
    resident waveform, and the stages after it have to survive being handed a manifest. ``after``
    names a stage to place it behind instead, and is still checked against both.
    """
    from nemo_curator.audio_agent import checkpoint as _checkpoint

    rec = _as_recipe(recipe).freeze()
    pviol = _safety.path_violations([output_path, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    spot, why = _checkpoint.advise(rec)
    if after:
        chosen = [i for i, s in enumerate(rec.stages) if s.ref == after]
        if not chosen:
            return {"status": "error", "reason": f"no stage named {after!r} in this recipe"}
        spot, why = _checkpoint.at(rec, index=chosen[-1] + 1)
    if spot is None:
        return {"status": "no_checkpoint", "reason": why, "advice": None}
    advice = spot.as_dict()
    if not output_path:
        return {"status": "advice", "advice": advice, "next": "call again with output_path to get the recipe"}
    checkpointed, err = _checkpoint.insert(rec, index=spot.index, output_path=output_path)
    if checkpointed is None:
        return {"status": "error", "reason": err, "advice": advice}
    return _safety.redact({
        "status": "ok",
        "advice": advice,
        "recipe": checkpointed.to_dict(),
        "next": "save this recipe, then validate -> smoke -> run it as usual",
    })


def delta_run(  # noqa: PLR0913 - the same execution knobs as run(), which it delegates to
    recipe: Recipe | dict[str, Any],
    *,
    data: str | None = None,
    confirm: bool | str = False,
    executor: Any = None,  # noqa: ANN401
    bootstrap_ray: bool = False,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
    goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the files that changed since a prior run, and merge them into its result.

    Without ``confirm`` this is a card: which files are new, changed or gone, which manifests
    would be rewritten, how many rows survive, and what it saves. With ``confirm`` it executes
    -- as an ordinary run of the user's own stages over the changed files (the source is
    narrowed by ``include_files``; no stage is told anything about being incremental), then
    merges each manifest and republishes it under the step key the full pipeline has for the
    enlarged corpus, so the next ordinary reuse probe finds it.

    Every question that has to be true first is answered by ``delta.plan`` and refused by name:
    which files moved, how deep per-file work stays independent of the other files, whether the
    stages' declarations survive their own recorded row counts, and which prior row came from
    which file. A refusal here is a full run, never a partial result presented as a whole one.
    """
    from nemo_curator.audio_agent import delta as _delta

    rec = _as_recipe(recipe).freeze()
    pviol = _safety.path_violations([data, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    binding = _dataset_binding(rec, data)
    dp = _profile_binding(binding)
    if dp is not None and _profile_error(dp):
        return _safety.redact(_profile_refusal(binding, dp))
    if _binding_blocks_execution(binding, data):
        return _safety.redact(_binding_refusal(binding))
    if dp is None:
        return {
            "status": "no_delta",
            "reason": "the source could not be profiled, so which files changed is unknown",
            "data_binding": binding.to_dict(),
        }

    decision = _delta.plan(
        rec,
        dataset_key=dp.dataset_key(),
        inventory=dp.inventory or None,
        inventory_root=dp.inventory_root,
    )
    card = {
        "status": "no_delta" if decision.status != "ready" else "ready",
        "recipe_id": rec.recipe_id,
        "config_hash": rec.config_hash,
        "delta": decision.to_dict(),
        "data_binding": binding.to_dict(),
    }
    if decision.status != "ready":
        card["next"] = "run the pipeline normally; a delta is not available for this input"
        return _safety.redact(card)
    # Same two-step shape ``run`` uses, for the same reason: gating on the hash comparison alone
    # is correct today (None/0/[]/{} all differ from it), but it is one edit away from not being.
    # ``run`` was hardened after a JSON-RPC ``confirm: null`` slipped through a single expression;
    # a security-relevant idiom should not have two spellings in one file.
    if confirm is not True and not (isinstance(confirm, str) and confirm == rec.config_hash):
        card.update(
            status="refused",
            reason=(
                "a delta run rewrites the prior manifests in place after merging; "
                "confirm it like any other run"
            ),
            confirm_with=f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed",
        )
        return _safety.redact(card)
    return _safety.redact(
        _execute_delta(
            rec,
            decision,
            dp=dp,
            data=data,
            card=card,
            executor=executor,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
        )
    )


def _carry_approval(
    approved: Recipe,
    derived: Recipe,
    *,
    confirm: bool | str,
    smoke_token: str | None,
) -> tuple[bool | str, str | None, str]:
    """Carry an approval given for ``approved`` onto the ``derived`` recipe that actually runs.

    Returns ``(confirm, smoke_token, refusal)``; a non-empty refusal means stop.

    Continuation and delta both execute a recipe THIS module derives -- a reader on the reused
    artifact plus the tail, or the prefix narrowed to the changed files. Its ``config_hash`` is
    one no human has ever seen, and holding the caller's evidence against it made both gates
    unsatisfiable rather than strict: ``continue --confirm <hash>`` (the form ``AGENTS.md``
    documents) was always refused, and under ``AUDIO_AGENT_REQUIRE_SMOKE`` no ``delta_run``
    could succeed at all, because the surface exposes no way to smoke a prefix recipe.

    The gate exists to bind INTENT. What the user read and approved is ``approved``; ``derived``
    is a deterministic function of it plus an artifact already verified against the registry, so
    the honest check is the human's evidence against the human's recipe -- after which the
    derived recipe may carry its own. This does not widen the gate: an unconfirmed call is left
    for ``run`` to refuse exactly as before, and a wrong hash or a missing/forged token still
    stops here.
    """
    # Not confirmed at all: let ``run`` answer, so its refusal keeps the scale estimate.
    if confirm is not True and not isinstance(confirm, str):
        return confirm, smoke_token, ""
    if isinstance(confirm, str) and confirm != approved.config_hash:
        return confirm, smoke_token, (
            "plan-execution integrity check failed: confirmed hash does not match the recipe"
        )
    if _safety.require_smoke() and not _safety.verify_smoke_token(smoke_token, approved.config_hash):
        return confirm, smoke_token, (
            "run requires smoke evidence (AUDIO_AGENT_REQUIRE_SMOKE is set): run smoke on this "
            "recipe and pass its 'smoke_token'"
        )
    carried = derived.config_hash if isinstance(confirm, str) else confirm
    return carried, _safety.smoke_token(derived.config_hash), ""


def _execute_delta(  # noqa: PLR0913 - forwards run()'s knobs and the plan it executes
    rec: Recipe,
    decision: Any,  # noqa: ANN401 - delta.Delta without an eager import
    *,
    dp: Any,  # noqa: ANN401 - DataProfile
    data: str | None,
    card: dict[str, Any],
    executor: Any,  # noqa: ANN401
    bootstrap_ray: bool,
    smoke_token: str | None,
    calibration: dict[str, Any] | None,
    goal: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run the changed files, merge, republish. Any failure leaves the prior result untouched."""
    from nemo_curator.audio_agent import delta as _delta

    sandbox = os.path.join(os.path.dirname(os.path.expanduser(decision.uri)), ".audio_agent_delta")
    prefix_rec, redirect, err = _delta.prefix_recipe(
        rec,
        prefix=decision.prefix,
        files=decision.files,
        sandbox=sandbox,
        sinks_=list(decision.sinks),
        inventory_key=getattr(dp, "inventory_key", "") or "",
    )
    if prefix_rec is None:
        return {**card, "status": "no_delta", "reason": err}
    # ``delta_run`` already held the caller's ``confirm`` to this recipe, so the approval is
    # established; what remains is to re-anchor it onto the prefix recipe that executes. Done
    # BEFORE the sandbox exists, so a refusal leaves nothing behind to clean up.
    inner_confirm, inner_token, refusal = _carry_approval(
        rec, prefix_rec, confirm=rec.config_hash, smoke_token=smoke_token
    )
    if refusal:
        return {**card, "status": "refused", "reason": refusal}
    try:
        os.makedirs(sandbox, exist_ok=True)
    except OSError as exc:
        return {**card, "status": "error", "reason": f"the delta's working directory could not be created: {exc}"}

    started = time.perf_counter()
    inner: dict[str, Any] = {"status": "completed", "note": "nothing to run: every changed file was removed"}
    if decision.files:
        inner = run(
            prefix_rec,
            confirm=inner_confirm,
            data=data,
            executor=executor,
            bootstrap_ray=bootstrap_ray,
            smoke_token=inner_token,
            calibration=calibration,
            goal=goal,
        )
        if inner.get("status") != "completed":
            return {**card, "status": "failed", "reason": "the delta's run did not complete", "run": inner}
    elapsed = round(time.perf_counter() - started, 3)

    # Every file the delta RAN, not just the ones whose prior rows went stale: an added file has
    # no prior rows, so this is a no-op on a first delta and an upsert on a retry. Without it, a
    # delta that failed on sink 2 of 3 appends sink 1's rows a second time when it is rerun.
    stale = set(decision.change.touched) | set(decision.change.removed)
    merges: list[dict[str, Any]] = []
    for sink in decision.sinks:
        kept, added, why = _delta.merge(
            sink,
            produced=redirect[sink.uri],
            stale=stale,
            key=sink.key,
            root=dp.inventory_root,
        )
        if why:
            # The failing sink's manifest is untouched, and any already merged hold a superset of
            # their prior rows. Those no longer match the digest their artifact was published
            # with, so the registry stops offering them ("serialized output changed after the
            # artifact was published") and the next run recomputes instead of serving a manifest
            # whose record understates it. Losing reuse is the safe direction; losing rows is not.
            return {**card, "status": "failed", "reason": why, "merged": merges}
        merges.append({**sink.summary(), "rows_kept": kept, "rows_added": added})

    published, problems = _delta.republish(
        rec,
        decision,
        dataset_key=dp.dataset_key(),
        fingerprint_tier=dp.fingerprint_tier,
        inventory=dict(dp.inventory),
        run_id=str(inner.get("run_id") or ""),
        added_sec=elapsed,
    )
    with contextlib.suppress(OSError):
        shutil.rmtree(sandbox)
    remaining = len(rec.stages) - decision.prefix
    result = {
        **card,
        # A prefix delta brings the checkpoint up to date, not the recipe's final output: the
        # stages past it still have to see every row. Reporting that as "completed" would present
        # a partial result as a whole one -- the one thing this verb promises never to do -- and a
        # host that reads the status and stops would hand the user yesterday's deliverable. The
        # status names the work that is left instead, and stays a success for the shell.
        "status": "completed" if remaining <= 0 else "tail_required",
        "ran_files": list(decision.files),
        "merged": merges,
        "published": published,
        "warnings": problems,
        "elapsed_sec": elapsed,
        "run": {k: inner.get(k) for k in ("status", "run_id", "output_paths") if k in inner},
        "next": (
            "the merged output covers every file; nothing further is needed"
            if remaining <= 0
            else f"the deliverable is NOT current yet: run the remaining {remaining} stage(s) with "
            "plan_continuation(execute=True, choice='extend') (CLI: continue --execute "
            "--choice extend --confirm), which now finds the merged manifest by an ordinary reuse probe"
        ),
    }
    if remaining > 0:
        result["tail"] = {
            "stages": remaining,
            "from_stage_index": decision.prefix,
            "stale_outputs": _stale_tail_outputs(rec, dp.dataset_key(), prefix=decision.prefix),
            "reason": decision.notes[0] if decision.notes else "the delta covers only a prefix of the recipe",
            # No acceptance verdict here on purpose: the criteria describe the recipe's final
            # output, which the tail has not produced yet. The continuation evaluates them.
            "acceptance": "judged after the remaining stage(s) run",
        }
    else:
        verdict = _delta_acceptance(rec, merged=merges, input_count=len(dp.inventory or {}))
        if verdict:
            result["acceptance"] = verdict
    return result


def _delta_acceptance(rec: Recipe, *, merged: list[dict[str, Any]], input_count: int) -> dict[str, Any]:
    """Judge the merged deliverable against the recipe's own success contract.

    A delta rewrites the user's output in place, so the bar they confirmed has to be re-checked
    against what is on disk NOW. Without this, ``status: completed`` says the curation finished
    while a ``must`` criterion it was gated on may be violated by the rows just merged in -- and
    the CLI exits 0, so a script ships it.

    Only called when the delta covered the whole recipe. For a prefix delta the final output does
    not exist yet, and a verdict computed over the checkpoint would be about a different artifact.
    """
    if not rec.acceptance_criteria or not merged:
        return {}
    deepest = merged[-1]
    rows = int(deepest.get("rows_kept") or 0) + int(deepest.get("rows_added") or 0)
    roles: list[str] = []
    keys: list[str] = []
    with contextlib.suppress(Exception):
        from nemo_curator.audio_agent.recipe import build_stages

        built, _ = build_stages(rec)
        roles, keys = _produced_roles_keys(built or [], None)
    return _acceptance_result(
        rec,
        types.SimpleNamespace(accepted=rows, input_count=input_count),
        roles,
        keys,
        [str(m.get("uri") or "") for m in merged if m.get("uri")],
    )


def _stale_tail_outputs(rec: Recipe, dataset_key: str, *, prefix: int) -> list[str]:
    """Where the tail stages write, and therefore what still describes the corpus before the delta.

    Named rather than left to the reader: "run the remaining stages" does not tell anyone which
    files on disk are currently a lie, and those are the ones a user is about to ship.
    """
    from nemo_curator.audio_agent import artifacts as art_mod

    with contextlib.suppress(Exception):
        return [p.uri for p in art_mod.plan_steps(rec, dataset_key)[prefix:] if p.persists() and p.uri]
    return []


def plan_continuation(  # noqa: PLR0913 - one verb covering plan + the three-way choice
    recipe: Recipe | dict[str, Any],
    parent_run_id: str | None = None,
    *,
    data: str | None = None,
    execute: bool = False,
    choice: str | None = None,
    confirm: bool | str = False,
    output_dir: str | None = None,
    checkpoint_path: str | None = None,
    bootstrap_ray: bool = False,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
    goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan — and optionally execute — a follow-up run that reuses prior work.

    Two engines feed one answer. A ``parent_run_id`` is diffed stage-by-stage against its
    :class:`RunRecord` (the honest, exact-append case), and the artifact registry is probed
    with this recipe's step keys, which additionally catches reuse the diff cannot see: a
    middle-of-pipeline edit, or work done by a *different* recipe that happens to share a
    prefix. The deeper of the two wins.

    With ``execute`` the plan stops being advice. ``choice`` picks the branch — ``as_is``
    (serve the existing output and re-verify the contract), ``extend`` (materialize a recipe
    that starts from the artifact and run only what is new), or ``fresh`` — defaulting to
    whatever the plan concluded. Execution still goes through the normal confirm gate.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. Omit it when the recipe is already unambiguous.
    """
    from nemo_curator.audio_agent import continuation, run_store
    from nemo_curator.audio_agent import reuse as _reuse

    rec = _as_recipe(recipe).freeze()
    pviol = _safety.path_violations(
        [data, output_dir, checkpoint_path, *_safety.recipe_path_params(rec)]
    )
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    binding = _dataset_binding(rec, data)
    dp = _profile_binding(binding)
    profile_error = _profile_error(dp) if dp is not None else ""
    binding_blocked = _binding_blocks_execution(binding, data) or bool(profile_error)
    dataset_key = dp.dataset_key() if dp and not binding_blocked else ""
    scan = _reuse.scan(rec, dataset_key=dataset_key)

    plan = _parent_plan(rec, parent_run_id, dp=dp, dataset_key=dataset_key)
    plan = _merge_plans(rec, plan, scan, dataset_key=dataset_key)
    plan["candidates"] = scan["candidates"]
    plan["estimated_saving_sec"] = scan["estimated_saving_sec"]
    plan["prompt_user"] = scan["prompt_user"]
    # Why the gate is asking, when it is not the size of the saving: work whose cost was never
    # recorded. Without this the card shows "saves 0.2 s" beside a question and looks broken.
    plan["unpriced_stages"] = scan.get("unpriced_stages", [])
    plan["recommended"] = scan.get("recommended", "fresh")
    plan["choices"] = scan.get("choices", [])
    plan["reuse_rationale"] = scan.get("rationale", "")
    plan["dataset_key"] = dataset_key
    plan["data_binding"] = binding.to_dict()
    if profile_error:
        plan["source_error"] = profile_error
        plan["reuse_rationale"] = f"prior work was not considered: {profile_error}"
    # Work that ran before but persisted nothing is invisible to BOTH engines: the parent diff
    # sees a changed recipe, the scan finds no artifact. Carry the disclosure onto the plan, or
    # the gate presents a recomputation as new work and the user pays for it twice unknowingly.
    if scan.get("prior_unsaved"):
        plan["prior_unsaved"] = scan["prior_unsaved"]
        plan["offer"] = scan["offer"]

    if not execute:
        return _safety.redact(plan)
    if binding_blocked:
        reason = profile_error or binding.reason
        return _safety.redact(
            {
                "status": "refused",
                "reason": f"continuation cannot execute with an unbound source: {reason}",
                "plan": plan,
                "data_binding": binding.to_dict(),
            }
        )
    return _execute_plan(
        rec,
        plan,
        choice=choice or _chosen_by_default(plan),
        data=data,
        confirm=confirm,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        bootstrap_ray=bootstrap_ray,
        smoke_token=smoke_token,
        calibration=calibration,
        goal=goal,
        parent=run_store.load(parent_run_id) if parent_run_id else None,
        continuation_mod=continuation,
    )


def _parent_plan(rec: Recipe, parent_run_id: str | None, *, dp: Any, dataset_key: str) -> dict[str, Any]:  # noqa: ANN401
    """The classic parent-diff plan, or a neutral 'nothing to diff against'."""
    from nemo_curator.audio_agent import continuation, run_store

    if not parent_run_id:
        return {"mode": "full_rerun", "reason": "no parent run given; reuse comes from the artifact scan alone",
                "run_stages": [s.ref for s in rec.stages]}
    parent = run_store.load(parent_run_id)
    if parent is None:
        return {"mode": "full_rerun", "reason": f"no parent run record {parent_run_id!r}; run fresh",
                "run_stages": [s.ref for s in rec.stages]}
    if dp is None or not dataset_key:
        return {
            "mode": "full_rerun",
            "parent_run_id": parent.run_id,
            "reason": "source data identity is unavailable; the parent cannot be claimed to be on the same data",
            "run_stages": [s.ref for s in rec.stages],
        }
    return continuation.plan_continuation(
        rec, parent, data_fingerprint=dp.fingerprint() if dp else None, dataset_key=dataset_key or None
    )


def _merge_plans(rec: Recipe, plan: dict[str, Any], scan: dict[str, Any], *, dataset_key: str) -> dict[str, Any]:
    """Take whichever engine reuses more VERIFIED work, and say which one found it.

    The parent diff and the step-key scan answer slightly different questions, so they can
    disagree, and depth breaks the tie. But depth alone used to decide it, and only the scan
    carried an artifact -- so a tie handed the plan to the engine that had proven nothing, and the
    executor then refused to extend for want of a resume URI. That was the ordinary case: a parent
    run whose artifact WAS published gives both engines the same depth.

    So the parent diff's claim is now resolved against the same registry, and verified depth beats
    unverified depth. When neither resolves, the plan says so instead of failing later.
    """
    from nemo_curator.audio_agent import reuse as _reuse

    scan_depth = len(scan.get("reuse_stages") or []) if scan["decision"] != "fresh" else 0
    plan_depth = len(plan.get("reuse_stages") or []) if plan["mode"] in ("already_done", "incremental") else 0
    if plan_depth >= scan_depth and plan_depth:
        point, why_not = _reuse.verified_point(rec, plan_depth, dataset_key=dataset_key)
        if point:
            plan["reuse_point"] = point
            plan.setdefault("reuse_from", [point["uri"]])
            plan.setdefault("source", "parent_diff")
            return plan
        if not scan_depth:
            plan["reuse_point_unavailable"] = why_not
            plan.setdefault("source", "parent_diff")
            return plan
        # The scan is shallower but backed by a real artifact, so it is the safer resume point.
        plan["superseded_parent_diff"] = f"deeper parent-diff reuse has no reusable artifact: {'; '.join(why_not)}"
    elif scan_depth <= plan_depth:
        plan.setdefault("source", "none")
        return plan
    point = scan["reuse_point"] or {}
    merged = {
        "mode": "already_done" if scan["decision"] == "already_done" else "incremental",
        "source": "artifact_scan",
        "parent_run_id": plan.get("parent_run_id"),
        "reuse_stages": scan["reuse_stages"],
        "run_stages": scan["run_stages"],
        "reuse_from": [point.get("uri")] if point.get("uri") else [],
        "reuse_point": point,
        "rationale": scan["rationale"],
    }
    superseded = plan.get("superseded_parent_diff") or plan.get("reason")
    if superseded:
        merged["superseded_parent_diff"] = superseded
    return merged


def _default_choice(mode: str) -> str:
    return {"already_done": "as_is", "incremental": "extend"}.get(mode, "fresh")


def _chosen_by_default(plan: dict[str, Any]) -> str:
    """What runs when the caller states no choice -- the plan's own recommendation if it made one.

    The scan already downgrades to ``fresh`` whenever a candidate's trust is anything less than
    high (a sampled dataset key that cannot see an in-place edit, a stage whose output is not
    reproducible). That recommendation only ever reached the approval card: execution went to the
    mode's default and reused the low-trust output anyway, so the warning and the behaviour
    disagreed. Recommending caution and then not taking it is worse than not warning at all.
    """
    recommended = str(plan.get("recommended") or "")
    offered = {str(c.get("id")) for c in plan.get("choices") or [] if isinstance(c, dict)}
    if recommended and (not offered or recommended in offered):
        return recommended
    return _default_choice(plan["mode"])


def _execute_plan(  # noqa: PLR0913 - executing a plan needs the plan's whole context
    rec: Recipe,
    plan: dict[str, Any],
    *,
    choice: str,
    data: str | None,
    confirm: bool | str,
    output_dir: str | None,
    bootstrap_ray: bool,
    goal: dict[str, Any] | None,
    parent: Any,  # noqa: ANN401
    continuation_mod: Any,  # noqa: ANN401
    checkpoint_path: str | None = None,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Carry out the user's three-way choice."""
    lineage = {"choice": choice, "mode": plan["mode"], "reuse_source": plan.get("source")}
    if choice == "fresh":
        return run(
            rec,
            confirm=confirm,
            data=data,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
            reuse={**lineage, "reused": []},
        )

    point = plan.get("reuse_point") or {}
    if choice == "as_is":
        return _serve_as_is(rec, plan, parent=parent, lineage=lineage)

    if plan["mode"] != "incremental" or not point.get("uri"):
        why = "; ".join(plan.get("reuse_point_unavailable") or []) or "the plan names no reusable output"
        return {
            "status": "refused",
            "reason": f"there is nothing to extend from ({why}); choose 'fresh'",
            "plan": plan,
        }

    materialized, err = continuation_mod.materialize(
        rec, uri=point["uri"], kind=point.get("kind", "unknown"), prefix=int(point.get("stage_index", -1)) + 1
    )
    if materialized is None:
        return {"status": "refused", "reason": err, "plan": plan}

    # Re-derive the boundary from the serialized artifact and its new source
    # stage. Cumulative in-memory metadata in the registry is not proof that a
    # manifest or bare audio directory serialized those roles/keys.
    verdict = validate(materialized, data=None)
    if not verdict.get("runnable", False):
        return _safety.redact(
            {
                "status": "refused",
                "reason": "the continued pipeline does not validate when resumed from the reused output",
                "recipe": materialized.to_dict(),
                "verdict": verdict,
                "plan": plan,
            },
            redact_transcripts=False,
        )
    inner_confirm, inner_token, refusal = _carry_approval(
        rec, materialized, confirm=confirm, smoke_token=smoke_token
    )
    if refusal:
        return _safety.redact(
            {
                "status": "refused",
                "reason": refusal,
                "confirm_with": f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed",
                "recipe": materialized.to_dict(),
                "plan": plan,
            },
            redact_transcripts=False,
        )
    result = run(
        materialized,
        confirm=inner_confirm,
        data=None,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        bootstrap_ray=bootstrap_ray,
        smoke_token=inner_token,
        calibration=calibration,
        goal=goal, reuse={**lineage, "reused": plan.get("reuse_stages", []), "reused_from": point.get("uri")},
        _continuation_context=_continuation_context_from_plan(rec, plan),
    )
    # An unconfirmed call is still refused by ``run`` (unchanged), but its advice names the
    # DERIVED hash -- which the user never saw and cannot meaningfully approve. Name theirs.
    if result.get("status") == "refused" and result.get("confirm_with"):
        result["confirm_with"] = f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed"
    result["recipe"] = _safety.redact(
        materialized.to_dict(),
        redact_transcripts=False,
    )
    result["plan"] = _safety.redact(plan, redact_transcripts=False)
    return _safety.redact(result)


def _declared_output(rec: Recipe) -> str:
    """The last output location the recipe names, or ``""`` -- where the user asked for it."""
    outs = _recipe_outputs(rec, None)
    return outs[-1] if outs else ""


def _deliver_to_declared_path(rec: Recipe, uri: str) -> tuple[str, bool | None]:
    """Put the reused output where the recipe asked for it.

    Returns ``(path, delivered)``: ``None`` when no copy was needed, ``True`` when the artifact
    was materialized at the declared path, ``False`` when it could not be.

    Serving the stored URI when the recipe declares a different ``output_path`` answers a
    question the user did not ask -- they get a path they never named, and the file they DID
    name is silently absent. Copying is cheap next to recomputing, so reuse honors the request.
    """
    want = _declared_output(rec)
    if not want or os.path.abspath(os.path.expanduser(want)) == os.path.abspath(os.path.expanduser(uri)):
        return uri, None
    src, dst = os.path.expanduser(uri), os.path.expanduser(want)
    try:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copyfile(src, dst)
    except OSError:
        return uri, False
    return want, True


def _has_content(uri: str) -> bool:
    """True if something is actually there -- a non-empty file, or a directory with entries."""
    path = os.path.expanduser(uri)
    if os.path.isdir(path):
        return bool(os.listdir(path))
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _as_is_evidence(
    plan: dict[str, Any],
    parent: Any,  # noqa: ANN401
    uri: str,
) -> tuple[str, list[str], int | None]:
    """How well the bytes about to be served are proven to be a complete result.

    Two engines can propose an ``as_is`` path and only one of them carries an artifact, so this is
    where both are held to a bar. The parent-diff path used to be served unconditionally from a
    recorded output path, which meant a record naming a file that was never written came back as
    ``status: reused`` with acceptance computed over nothing at all.

    An artifact is the strong evidence: a ``_COMPLETE`` marker, a matching dataset and code
    version. Without one -- a run from before artifacts, or a pruned record -- fall back to what
    can still be checked rather than refusing outright, and name the weaker basis in the result.
    """
    from nemo_curator.audio_agent import artifacts as art_mod

    point = plan.get("reuse_point") or {}
    if point.get("step_key"):
        art, reasons = art_mod.lookup(point["step_key"], dataset_key=plan.get("dataset_key") or None)
        if art is not None and not reasons:
            marker = art_mod.read_marker(art.uri) or {}
            artifact_rows = getattr(art, "rows_out", None)
            marker_rows = marker.get("rows")
            expected_rows = (
                artifact_rows
                if (
                    isinstance(artifact_rows, int)
                    and not isinstance(artifact_rows, bool)
                    and artifact_rows >= 0
                    and marker_rows == artifact_rows
                )
                else None
            )
            return "artifact", [], expected_rows
        return "", reasons or ["the recorded artifact is no longer reusable"], None
    status = str(getattr(parent, "status", "") or "")
    if status and status != "completed":
        return "", [
            f"the run that produced it ended {status!r}, so its output is not a complete result"
        ], None
    if not _has_content(uri):
        return "", [f"{uri!r} does not exist or is empty, so there is nothing to serve"], None
    return "run_record", [], None


def _serve_as_is(rec: Recipe, plan: dict[str, Any], *, parent: Any, lineage: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """Serve an existing output — and RE-VERIFY it against the criteria asked for now.

    Reused data is still judged by today's bar: a stricter contract must be re-checked, never
    inherited from the run that produced the bytes.
    """
    point = plan.get("reuse_point") or {}
    uri = point.get("uri") or next(iter(plan.get("reuse_from") or []), None)
    if not uri:
        detail = str(plan.get("reuse_rationale") or "").strip()
        reason = "no completed output can be served"
        if detail:
            reason += f": {detail}"
        return {"status": "refused", "reason": f"{reason}; choose 'fresh'", "plan": plan}
    pviol = _safety.path_violations([uri, _declared_output(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "reused input/output resolves outside the allowed workspace",
            "violations": pviol,
            "plan": plan,
        }
    # Checked BEFORE delivery: a refusal must not leave a copy behind.
    trust, why_not, expected_output_rows = _as_is_evidence(plan, parent, uri)
    if not trust:
        return {
            "status": "refused",
            "reason": f"the previous output cannot be served: {'; '.join(why_not)}; choose 'fresh'",
            "plan": plan,
        }
    uri, delivered_to = _deliver_to_declared_path(rec, uri)
    if delivered_to is False:
        return {
            "status": "refused",
            "reason": (
                f"the recipe asks for output at {_declared_output(rec)!r} but the reusable output is at "
                f"{uri!r}, and it could not be copied there; choose 'fresh' or point the recipe at that path"
            ),
            "plan": plan,
        }
    per_item, output_scan = (
        _scan_terminal_output([uri], limit=0)
        if _needs_terminal_evidence(rec)
        else ([], {})
    )
    source_run = parent
    source_run_id = getattr(source_run, "run_id", None)
    if source_run is None:
        source_run_id = point.get("run_id") or (
            (plan.get("candidates") or [{}])[0].get("run_id")
        )
        if source_run_id:
            from nemo_curator.audio_agent import run_store

            source_run = run_store.load(str(source_run_id))
    source_input_count = getattr(source_run, "input_count", None)
    if (
        not isinstance(source_input_count, int)
        or isinstance(source_input_count, bool)
        or source_input_count < 0
    ):
        source_input_count = None
    acceptance = verify(
        list(rec.acceptance_criteria),
        {
            "produced_roles": list(point.get("produced_roles") or []),
            "produced_keys": list(point.get("produced_keys") or []),
            # The artifact is a real file, so judge today's bar against its rows -- reused data
            # gets the same data-level scrutiny a fresh run gets, not a weaker label check.
            "per_item": per_item,
            "output_scan": output_scan,
            "metrics": _aggregate_metrics(output_scan),
            # A valid artifact binds this count in both its registry record and
            # completion marker. A legacy run record names a path but does not
            # independently prove its serialized inventory, so it stays unknown.
            "expected_output_rows": expected_output_rows,
            "retained": int(point.get("rows") or 0),
            # Relative yield is defined against the producing run's source
            # inventory.  Substituting output rows here manufactures 100%
            # retention whenever the run record is unavailable.
            "input_count": source_input_count,
        },
    )
    checked = len(rec.acceptance_criteria)
    note = "served from a previously completed run; no compute was done. "
    note += f"The {checked} success criterion(a) were re-checked." if checked else "No success criteria were supplied, so nothing was re-checked."
    if delivered_to:
        note += f" The reused output was copied to the path the recipe asked for ({uri})."
    if trust == "run_record":
        note += (
            " No artifact record backs this output, so completeness rests on the producing run"
            " having finished rather than on a completion marker."
        )
    return _safety.redact(
        {
            "status": "reused",
            "output": uri,
            "rows": point.get("rows") if point.get("rows") is not None else _row_count(uri),
            "evidence": trust,
            "reuse": {**lineage, "reused": plan.get("reuse_stages", []), "reused_from": uri},
            "acceptance": acceptance,
            "source_run_id": source_run_id,
            "note": note,
            "plan": plan,
        }
    )


def calibrate(smoke_report: dict[str, Any]) -> dict[str, Any]:
    """Extract measured per-stage resources from a smoke report (1C.2).

    Returns ``{calibration: {stage: {gpu_mem_gb?, host_mem_gb?, ...}}}`` to pass to
    ``run(..., calibration=...)`` so the planner can raise card ``best_guess``
    facts when the smoke observed a larger peak. A bounded smoke never lowers a
    card/default estimate. A CPU smoke still measures host RAM and throughput (it
    only lacks VRAM), and host RAM is what decides streaming vs batch.

    Calling this is optional: ``smoke`` already stores its measurements under the
    recipe's ``config_hash`` and ``run`` applies them when no calibration is passed.
    Extract them here to inspect, archive, or hand a run measurements it would not
    otherwise find (e.g. from a differently sized machine).
    """
    from nemo_curator.audio_agent import calibration as _cal

    existing = smoke_report.get("calibration")
    if isinstance(existing, dict):
        # ``smoke`` already extracted and machine-stamped these measurements.
        # Re-extracting from the surrounding report silently discarded that
        # binding and let measurements migrate to another machine.
        result: dict[str, Any] = {"calibration": existing}
        wrapper_fingerprint = smoke_report.get("machine_fingerprint")
        if not wrapper_fingerprint:
            fingerprints = {
                entry.get("machine_fingerprint")
                for entry in existing.values()
                if isinstance(entry, dict) and entry.get("machine_fingerprint")
            }
            if len(fingerprints) == 1:
                wrapper_fingerprint = fingerprints.pop()
        if isinstance(wrapper_fingerprint, str) and wrapper_fingerprint:
            result["machine_fingerprint"] = wrapper_fingerprint
        return result

    machine_fingerprint = smoke_report.get("machine_fingerprint")
    return {
        **(
            {"machine_fingerprint": machine_fingerprint}
            if isinstance(machine_fingerprint, str) and machine_fingerprint
            else {}
        ),
        "calibration": _cal.from_smoke(
            smoke_report,
            machine_fingerprint=(
                machine_fingerprint
                if isinstance(machine_fingerprint, str)
                else None
            ),
        ),
    }


# --------------------------------------------------------------------------- #
# host instructions (skills)
# --------------------------------------------------------------------------- #
# Where each host looks for a project- and user-level skill. Codex scans ``.agents/skills``
# from the CWD up to the repo root and Cursor lists it as a project discovery root, so both
# share one directory; Claude Code reads only ``.claude/skills`` and needs its own entry.
_SKILL_HOST_DIRS: dict[str, tuple[str, str]] = {
    "codex": (".agents/skills", "~/.agents/skills"),
    "cursor": (".agents/skills", "~/.cursor/skills"),
    "claude": (".claude/skills", "~/.claude/skills"),
}


def skills_dir() -> str:
    """The packaged skill definitions — the single source of truth every host reads.

    Ships inside the wheel (``MANIFEST.in`` takes ``nemo_curator/**/*.md``), so a
    ``pip install`` user has the same instructions a checkout does, without ``.claude/``
    or ``.cursor/`` which live outside the package.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def available_skills() -> list[str]:
    """Packaged skill names, i.e. the directories under :func:`skills_dir` holding a SKILL.md."""
    root = skills_dir()
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, "SKILL.md"))
    )


def _trees_match(source: str, target: str) -> bool:
    """Whether ``target`` already holds a byte-identical copy of everything in ``source``.

    Lets a repeated install report ``unchanged`` instead of rewriting files, and lets a
    genuinely different directory of the same name be reported as a conflict rather than
    silently overwritten.
    """
    for dirpath, _dirnames, filenames in os.walk(source):
        rel = os.path.relpath(dirpath, source)
        mirror = target if rel == "." else os.path.join(target, rel)
        if not os.path.isdir(mirror):
            return False
        for filename in filenames:
            dst_file = os.path.join(mirror, filename)
            if not os.path.isfile(dst_file):
                return False
            try:
                if not filecmp.cmp(os.path.join(dirpath, filename), dst_file, shallow=False):
                    return False
            except OSError:
                return False
    return True


def _is_symlink_farm(source: str, target: str) -> bool:
    """Whether ``target`` is a real directory whose entries are links to ``source``'s.

    This is the shape a symlink install leaves behind, so recognizing it is what makes a
    second install report ``unchanged`` rather than relaying identical links.
    """
    if not os.path.isdir(target) or os.path.islink(target):
        return False
    entries = os.listdir(source)
    if sorted(entries) != sorted(os.listdir(target)):
        return False
    return all(
        os.path.islink(os.path.join(target, entry))
        and os.path.realpath(os.path.join(target, entry)) == os.path.realpath(os.path.join(source, entry))
        for entry in entries
    )


def _names_its_own_target(path: str, source_entry: str) -> bool:
    """A plain file whose whole content is the path it was supposed to link to."""
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    try:
        if os.path.getsize(path) > 4096:  # noqa: PLR2004 - a path, not a document
            return False
        with open(path, encoding="utf-8", errors="strict") as handle:
            text = handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return False
    return bool(text) and "\n" not in text and os.path.basename(text) == os.path.basename(source_entry)


def _is_dead_shim(target: str, source: str) -> bool:
    """A shim git recorded but could not materialize, so it is text where a link belongs.

    On a Windows checkout without ``core.symlinks`` every committed link becomes a file
    holding its target path, and every host silently finds no skill. Replacing that is a
    repair rather than a destructive overwrite, so it does not require ``force``.
    """
    if os.path.isdir(target) and not os.path.islink(target):
        entries = os.listdir(source)
        return bool(entries) and all(
            _names_its_own_target(os.path.join(target, entry), os.path.join(source, entry))
            for entry in entries
        )
    return _names_its_own_target(target, source)


def _existing_install(source: str, target: str) -> str | None:
    """What is already sitting at ``target``, in terms this installer can act on.

    ``links`` and ``copy`` both carry exactly the source's content, so switching between
    them is a relayout rather than a loss. Anything else is somebody's own work.
    """
    if not os.path.lexists(target):
        return None
    if _is_symlink_farm(source, target):
        return "links"
    if os.path.islink(target) and os.path.realpath(target) == os.path.realpath(source):
        return "whole_dir_link"  # an older install, or a hand-made shim
    if _is_dead_shim(target, source):
        return "dead"
    if os.path.isdir(target) and not os.path.islink(target) and _trees_match(source, target):
        return "copy"
    return "foreign"


def _install_action(source: str, target: str, *, mode: str, force: bool) -> str:
    """What installing ``source`` at ``target`` would do, without doing any of it."""
    found = _existing_install(source, target)
    if found is None:
        return "created"
    if found == "dead":
        return "repaired"
    if found == "foreign":
        # A hand-written skill, or an install that has since been edited. Overwriting it
        # would delete work the caller never mentioned.
        return "replaced" if force else "conflict"
    already_right = ("links" if mode == "symlink" else "copy") == found
    return "unchanged" if already_right else "replaced"


def _lay_out_skill(source: str, target: str, *, mode: str) -> None:
    """Write the skill at ``target``, replacing whatever is there.

    A symlink install links each top-level entry into a real directory rather than linking
    the directory itself. Both shapes resolve identically when read, but a walker that does
    not follow symlinks -- a common default, and what a ripgrep-backed file search does --
    lists nothing at all inside a symlinked directory, so the skill is silently undiscovered.
    A symlinked file inside a real directory is an ordinary entry to such a walk.
    """
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.lexists(target):
        if os.path.islink(target) or os.path.isfile(target):
            os.unlink(target)
        else:
            shutil.rmtree(target)
    if mode != "symlink":
        shutil.copytree(source, target)
        return
    os.makedirs(target)
    for entry in os.listdir(source):
        entry_source = os.path.join(source, entry)
        # Absolute: an installed source sits in site-packages, arbitrarily far from the
        # destination, so a relative link would be long and break if either side moved.
        os.symlink(
            os.path.abspath(entry_source),
            os.path.join(target, entry),
            target_is_directory=os.path.isdir(entry_source),
        )


def _install_one_skill(
    source: str, target: str, *, mode: str, force: bool, dry_run: bool
) -> dict[str, Any]:
    """Place one skill at ``target``, refusing to destroy anything that is not ours."""
    action = _install_action(source, target, mode=mode, force=force)
    entry: dict[str, Any] = {"target": target, "mode": mode, "action": action}
    if action == "conflict":
        entry["reason"] = (
            "already exists with different content; pass force to replace it, "
            "or install under a different scope"
        )
        return entry
    if action == "unchanged" or dry_run:
        return entry

    try:
        _lay_out_skill(source, target, mode=mode)
    except OSError as exc:
        entry["action"] = "error"
        entry["reason"] = str(exc)
        if mode == "symlink":
            # The usual cause is Windows without developer mode or admin rights.
            entry["hint"] = "symlinks may be unavailable on this platform; retry with copy mode"
    return entry


def _install_request_error(scope: str, host: str, mode: str, dest: str | None, skills: list[str] | None) -> str | None:
    """The first thing wrong with an install request, or ``None`` if it is coherent."""
    if scope not in {"project", "user"}:
        return f"scope must be 'project' or 'user', got {scope!r}"
    if mode not in {"copy", "symlink"}:
        return f"mode must be 'copy' or 'symlink', got {mode!r}"
    unknown = [h for h in ([] if host == "all" else [host]) if h not in _SKILL_HOST_DIRS]
    if unknown:
        return f"unknown host(s) {unknown!r}; choose from {sorted(_SKILL_HOST_DIRS)} or 'all'"
    if dest and scope == "user":
        return "dest applies to project scope; user scope installs under the home directory"
    packaged = available_skills()
    missing = [name for name in skills or [] if name not in packaged]
    if missing:
        return f"no packaged skill named {missing!r}; available: {packaged}"
    return None


def install_skill(  # noqa: PLR0913 - one verb covering scope, host, mode and the safety flags
    *,
    scope: str = "project",
    host: str = "all",
    mode: str = "copy",
    skills: list[str] | None = None,
    dest: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install the packaged audio skills into the directories the hosts discover.

    The instructions live in the wheel, but no host looks inside site-packages: Codex and
    Cursor read ``.agents/skills``, Claude Code reads ``.claude/skills``. A checkout has
    those wired up already; this is how a ``pip install`` user gets the same workflow.

    ``scope`` is ``project`` (the CWD, or ``dest``) or ``user`` (the home-directory
    equivalents). ``mode`` defaults to ``copy`` because it works everywhere, including a
    Windows checkout without symlink support; ``symlink`` keeps an installed skill in step
    with the package and is what a developer wants. Idempotent, and refuses to replace a
    directory whose content differs unless ``force`` is set.
    """
    bad_request = _install_request_error(scope, host, mode, dest, skills)
    if bad_request:
        return {"status": "error", "error": bad_request}

    source_root = skills_dir()
    wanted = skills or available_skills()
    hosts = sorted(_SKILL_HOST_DIRS) if host == "all" else [host]

    # One directory may serve several hosts (Codex and Cursor share ``.agents/skills``),
    # so collapse them first: installing twice would report the second pass as unchanged
    # and make the summary read as if more was written than actually was.
    roots: dict[str, list[str]] = {}
    for name in hosts:
        project_dir, user_dir = _SKILL_HOST_DIRS[name]
        if scope == "project":
            root = os.path.abspath(os.path.join(dest or os.getcwd(), project_dir))
        else:
            root = os.path.abspath(os.path.expanduser(user_dir))
        roots.setdefault(root, []).append(name)

    targets = [os.path.join(root, name) for root in roots for name in wanted]
    pviol = _safety.path_violations(targets)
    if pviol:
        return {
            "status": "refused",
            "reason": "target path(s) resolve outside the allowed workspace",
            "violations": pviol,
            "hint": (
                "AUDIO_AGENT_WORKSPACE confines writes to that tree; install into the "
                "workspace with project scope, or unset the lock to install for the user"
            ),
        }

    installed: list[dict[str, Any]] = []
    for root, host_names in roots.items():
        for name in wanted:
            entry = _install_one_skill(
                os.path.join(source_root, name),
                os.path.join(root, name),
                mode=mode,
                force=force,
                dry_run=dry_run,
            )
            installed.append({"skill": name, "hosts": host_names, **entry})

    failed = [e for e in installed if e["action"] in {"conflict", "error"}]
    return {
        "status": "error" if failed else "ok",
        "scope": scope,
        "mode": mode,
        "dry_run": dry_run,
        "source": source_root,
        "installed": installed,
        **({"unresolved": failed} if failed else {}),
        "next_step": (
            "restart the host (or reload its skills) so it rescans the directory"
            if not dry_run and not failed
            else "nothing was written"
            if dry_run
            else "resolve the conflicts above, or re-run with force"
        ),
    }


# --------------------------------------------------------------------------- #
# execution + bounding helpers
# --------------------------------------------------------------------------- #
def _bootstrap_ray() -> str:
    """Ensure a Ray cluster (opt-in) and return its address; sets RAY_ADDRESS."""
    from nemo_curator.audio_agent._ray import ensure_cluster

    return ensure_cluster()


def _shutdown_owned_ray(address: str | None) -> bool | None:
    """Stop only a local Ray head this process created.

    ``None`` means the address was external/not owned, ``True`` means cleanup
    completed, and ``False`` means an owned head could not be stopped safely.
    """
    if not address:
        return None
    from nemo_curator.audio_agent._ray import owns_cluster, shutdown_cluster

    if not owns_cluster(address):
        return None
    return shutdown_cluster(address)


def _resource_environment(
    env: Any,  # noqa: ANN401
    address: str | None,
    execution_target: str,
) -> Any:  # noqa: ANN401
    """Return only capacity facts that belong to the selected executor."""
    if execution_target != "custom_executor":
        return _apply_ray_cluster_capacity(env, address)
    target_env = copy.deepcopy(env)
    target_env.has_gpu = False
    target_env.gpu_count = 0
    target_env.gpu_names = []
    target_env.gpu_mem_gb = 0.0
    target_env.gpu_visibility = "unknown"
    target_env.total_cpus = 0
    target_env.total_ram_gb = 0.0
    target_env.free_disk_gb = None
    target_env.notes = list(getattr(target_env, "notes", []) or [])
    target_env.notes.append(
        "custom executor capacity is caller-owned and unverified; driver capacity was not substituted"
    )
    return target_env


def _adapt_resource_plan_for_target(
    plan: Any,  # noqa: ANN401
    *,
    execution_target: str,
    operation: str,
) -> None:
    """Keep target uncertainty explicit without making bounded verification impossible."""
    if not isinstance(getattr(plan, "notes", None), list):
        plan.notes = []
    escalations = list(getattr(plan, "escalations", []) or [])
    if execution_target == "custom_executor":
        if escalations:
            plan.notes.append(
                "custom_executor_unverified_capacity=" + "; ".join(escalations)
            )
        plan.escalations = []
        plan.feasible = True
        plan.notes.append(
            "custom executor owns scheduling; capacity is verified by its bounded execution"
        )
        return
    # Dormant by design, not by oversight: the planner currently reports unknown VRAM as a
    # NOTE and never escalates on it ("VRAM is intentionally NOT escalated here" --
    # ``planner.py``), so no live plan reaches this with such an escalation. It stays because
    # the rule it encodes outlives that choice: a bounded smoke is how VRAM gets MEASURED on a
    # remote cluster, so refusing the smoke for want of the measurement would close the only
    # door to it. A run is a different matter and still escalates.
    unknown_vram = [
        item
        for item in escalations
        if item.startswith("GPU VRAM capacity is unknown;")
    ]
    if (
        execution_target == "external_ray"
        and operation == "smoke"
        and escalations
        and len(unknown_vram) == len(escalations)
    ):
        plan.notes.extend(
            "bounded_remote_smoke=" + item
            for item in unknown_vram
        )
        plan.escalations = []
        plan.feasible = True


def _apply_ray_cluster_capacity(env: Any, address: str | None) -> Any:  # noqa: ANN401
    """Overlay driver probes with the resources the executor can schedule."""
    if not address:
        return env
    try:
        from nemo_curator.audio_agent._ray import cluster_resources

        resources = cluster_resources(address)
    except Exception as exc:  # noqa: BLE001 - an unprobed executor must fail closed
        raise RuntimeError(
            "Ray cluster capacity could not be verified; refusing to substitute "
            f"driver resources ({type(exc).__name__}: {exc})"
        ) from exc

    cpus = float(resources.get("CPU", 0.0) or 0.0)
    gpus = float(resources.get("GPU", 0.0) or 0.0)
    memory = float(resources.get("memory", 0.0) or 0.0)
    if not math.isfinite(cpus) or cpus <= 0:
        raise RuntimeError(
            f"Ray cluster at {address!r} reported no positive finite CPU capacity"
        )
    if not math.isfinite(gpus) or gpus < 0:
        raise RuntimeError(
            f"Ray cluster at {address!r} reported invalid GPU capacity {gpus!r}"
        )
    if not math.isfinite(memory) or memory < 0:
        raise RuntimeError(
            f"Ray cluster at {address!r} reported invalid memory capacity {memory!r}"
        )
    env.total_cpus = cpus
    env.gpu_count = max(0, int(gpus))
    env.has_gpu = env.gpu_count > 0
    if memory > 0:
        env.total_ram_gb = round(memory / (1024**3), 1)

    if not _ray_address_is_local(address):
        # Ray exposes GPU count, not per-device VRAM/name. Driver GPU details
        # must not be projected onto a remote or unresolved cluster.
        env.gpu_mem_gb = 0.0
        env.gpu_names = []
    env.notes.append(
        "resource planning bound to Ray cluster capacity at "
        f"{address}: CPU={cpus:g}, GPU={gpus:g}"
    )
    return env


def _run_pipeline(stages: list[Any], executor: Any, *, checkpoint_path: str | None = None) -> list[Any] | None:  # noqa: ANN401
    from nemo_curator.pipeline import Pipeline

    pipeline = Pipeline(name="audio_agent_run", stages=list(stages))
    # Keep the verb's stdout pure JSON: backend/worker logs (Ray forwards them to the
    # driver's stdout, e.g. verbose NeMo output) go to stderr during execution, so a
    # host parsing the CLI's stdout never sees them interleaved with the result.
    with contextlib.redirect_stdout(sys.stderr):
        return pipeline.run(executor, checkpoint_path=checkpoint_path)


def _calibration_for_run(
    explicit: dict[str, Any] | None,
    config_hash: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Pick the measurements to plan a run with, and say where they came from.

    Returns ``(calibration, provenance_note)``. What the caller passed always wins: the store
    is a fallback for the flag nobody remembered, never an override of the one they did pass.
    The note records the substitution on the resource plan, because a run must never be
    planned from evidence its operator cannot see. Whether each stored entry is actually
    *applied* stays the planner's call -- it drops any measurement stamped with a different
    machine, and says so in its own notes.
    """
    if explicit is not None:
        return explicit, None
    from nemo_curator.audio_agent import calibration_store

    stored = calibration_store.load(config_hash)
    if not stored:
        return None, None
    measured_at = stored.get("created_at")
    return stored, (
        "calibration: none passed; using the per-stage measurements a prior smoke of this "
        + f"exact recipe stored{f' at {measured_at}' if measured_at else ''} "
        + "(pass --calibration to override)"
    )


def _plan_resources(stages: list[Any], env_obj: Any, data_profile: dict[str, Any] | None, calibration: dict[str, Any] | None = None):  # noqa: ANN401
    """Run the deterministic resource planner over the built stages (1C.1 + 1C.2 calibration)."""
    from nemo_curator.audio_agent import planner
    from nemo_curator.stages.audio import agent as foundation

    # Release GPU reservations held by gpu-OPTIONAL stages when this host has none, so a
    # supported CPU-only (audio_cpu) install plans and runs instead of being refused.
    # Mutates `stages` in place, which is the same list the caller then executes.
    cpu_notes = planner.cpu_fallback(stages, env_obj)
    contracts: list[Any] = []
    for st in stages:
        try:
            contracts.append(foundation.build_contract(st))
        except Exception:  # noqa: BLE001 - a stage that can't describe itself gets conservative defaults
            contracts.append(None)
    rplan = planner.plan(stages, contracts, env_obj, data_profile, calibration=calibration)
    for note in cpu_notes:
        rplan.notes.append("cpu_fallback: " + note)
    return rplan


def _make_executor(mode: str) -> Any:  # noqa: ANN401
    """A XennaExecutor with the planned execution_mode; None -> the pipeline default."""
    try:
        from nemo_curator.backends.xenna import XennaExecutor

        return XennaExecutor({"execution_mode": mode})
    except Exception:  # noqa: BLE001 - fall back to the pipeline's default executor (streaming)
        return None


def _bound_recipe(
    recipe: Recipe,
    sample: int,
    rpt: SmokeReport,
    binding: Any,  # noqa: ANN401 - DatasetBinding without an eager import
) -> _SmokeBound:
    """Prove that the source can emit at most ``sample`` rows.

    Bounding is a closed adapter table, just like source identity. A source that
    cannot be capped without downloading or reading an unknown remote selector is
    refused before stage construction. The original recipe and every stage default
    remain untouched.
    """
    bounded = copy.deepcopy(recipe)
    if not bounded.stages:
        return _SmokeBound(None, error="recipe has no source stage")
    src = bounded.stages[0]
    if src.ref == "ManifestReader":
        manifests = tuple(getattr(binding, "selected_manifest_files", ()) or ())
        if not manifests:
            return _SmokeBound(
                None,
                error=(
                    "ManifestReader selectors are not a complete ordered set of "
                    "local manifest files; remote or mixed selectors are unsupported"
                ),
            )
        tmp, count, error = _write_bounded_manifest(manifests, sample)
        if error:
            return _SmokeBound(None, error=error)
        execution_knobs = {
            key: value for key, value in src.params.items() if key in EXECUTION_KNOB_PARAMS
        }
        src.params = {
            "manifest_path": tmp,
            "files_per_partition": 1,
            "blocksize": None,
            "file_extensions": [".jsonl"],
            "storage_options": None,
            **execution_knobs,
        }
        rpt.notes.append(
            f"bounded via {len(manifests)} resolved local manifest file(s) "
            f"({count}/{sample} rows)"
        )
        return _SmokeBound(bounded, (tmp,), count)

    if src.ref == "ReadLongFormManifestStage":
        manifest = getattr(binding, "primary_path", None)
        if not isinstance(manifest, str):
            return _SmokeBound(None, error="long-form manifest path was not resolved")
        tmp, count, error = _write_bounded_manifest((manifest,), sample)
        if error:
            return _SmokeBound(None, error=error)
        src.params["input_manifest"] = tmp
        rpt.notes.append(f"bounded via truncated long-form manifest ({count}/{sample} rows)")
        return _SmokeBound(bounded, (tmp,), count)

    if src.ref == "CreateInitialManifestFleursStage":
        if getattr(binding, "generated", False) or not getattr(binding, "profile_source", None):
            return _SmokeBound(
                None,
                error=(
                    "unstaged FLEURS would download and materialize the complete split; "
                    "pre-stage the dataset before smoke"
                ),
            )
        tmp, count, error = _write_bounded_fleurs_manifest(src, binding, sample)
        if error:
            return _SmokeBound(None, error=error)
        execution_knobs = {
            key: value for key, value in src.params.items() if key in EXECUTION_KNOB_PARAMS
        }
        src.ref = "ManifestReader"
        src.params = {"manifest_path": tmp, **execution_knobs}
        rpt.notes.append(f"bounded pre-staged FLEURS via temporary manifest ({count}/{sample} rows)")
        return _SmokeBound(bounded, (tmp,), count)

    if src.ref == "CreateInitialManifestReadSpeechStage" and getattr(binding, "generated", False):
        return _SmokeBound(
            None,
            error=(
                "unstaged ReadSpeech would download and extract the complete archive; "
                "pre-stage the dataset before smoke"
            ),
        )

    if src.ref in {
        "CreateInitialManifestAudioFolderStage",
        "CreateInitialManifestReadSpeechStage",
    }:
        src.params["max_samples"] = sample
        rpt.notes.append(f"bounded via max_samples={sample}")
        return _SmokeBound(bounded)

    return _SmokeBound(
        None,
        error=f"source stage {src.ref!r} has no proven smoke-bounding adapter",
    )


def _isolate_smoke_outputs(bound: _SmokeBound, rpt: SmokeReport) -> _SmokeBound:
    """Redirect every reviewed output mutation into one disposable local tree."""
    if bound.recipe is None:
        return bound
    try:
        root = tempfile.mkdtemp(prefix="audio_agent_smoke_outputs_")
    except OSError as exc:
        return _SmokeBound(
            None,
            tmp_paths=bound.tmp_paths,
            input_count=bound.input_count,
            error=f"temporary output sandbox could not be created: {type(exc).__name__}: {exc}",
        )

    aliases: dict[str, str] = {}
    try:
        for index, stage in enumerate(bound.recipe.stages):
            for key in sorted(_SMOKE_FILE_OUTPUT_PARAMS | _SMOKE_DIR_OUTPUT_PARAMS):
                value = stage.params.get(key)
                if not isinstance(value, str) or not value:
                    continue
                stage.params[key] = _smoke_output_path(
                    root,
                    index,
                    stage.ref,
                    key,
                    value,
                    aliases,
                )

            if (
                stage.ref == "MonoConversionStage"
                and stage.params.get("write_to_disk", False)
                and not stage.params.get("output_dir")
            ):
                stage.params["output_dir"] = _smoke_output_path(
                    root,
                    index,
                    stage.ref,
                    "output_dir",
                    f"<implicit:{index}:output_dir>",
                    aliases,
                )

            if stage.ref == "PyAnnoteDiarizationStage" and stage.params.get(
                "write_rttm",
                True,
            ):
                stage.params["write_rttm"] = False
                rpt.notes.append(
                    "disabled source-adjacent PyAnnote RTTM side output for smoke"
                )

            if stage.ref in {"SplitLongAudioStage", "SplitASRAlignJoinStage"}:
                stage.params["output_dir"] = _smoke_output_path(
                    root,
                    index,
                    stage.ref,
                    "output_dir",
                    f"<implicit:{index}:split_output_dir>",
                    aliases,
                )

            if stage.ref == "CreateInitialManifestReadSpeechStage":
                # Binding already proved this source is staged. Prevent a
                # disappearing/racing directory from turning smoke into a 4.88 GB
                # acquisition between the read-only check and stage execution.
                stage.params["auto_download"] = False
    except (OSError, TypeError, ValueError) as exc:
        _cleanup([root])
        return _SmokeBound(
            None,
            tmp_paths=bound.tmp_paths,
            input_count=bound.input_count,
            error=f"output sandbox could not be configured: {type(exc).__name__}: {exc}",
        )

    rpt.notes.append("all pipeline outputs isolated in temporary smoke storage")
    return _SmokeBound(
        bound.recipe,
        (*bound.tmp_paths, root),
        bound.input_count,
        output_root=root,
    )


def _smoke_output_path(
    root: str,
    stage_index: int,
    stage_ref: str,
    key: str,
    original: str,
    aliases: dict[str, str],
) -> str:
    """Map a production location to a stable path inside ``root``."""
    if original in aliases:
        return aliases[original]
    stage_dir = os.path.join(root, f"{stage_index:03d}_{stage_ref}")
    if key in _SMOKE_FILE_OUTPUT_PARAMS:
        from urllib.parse import urlsplit

        basename = os.path.basename(urlsplit(original).path.rstrip("/"))
        filename = f"{key}_{basename or 'output'}"
        target = os.path.join(stage_dir, filename)
    else:
        target = os.path.join(stage_dir, key)
    aliases[original] = target
    return target


def _smoke_write_issues(
    stages: list[Any],
    output_root: str | None,
) -> list[str]:
    """Find a disk-writing stage whose output is not proved to be isolated."""
    if not output_root:
        return ["temporary output root is missing"]

    from nemo_curator.stages.audio import agent as foundation

    issues: list[str] = []
    for stage in _walk_smoke_stages(stages):
        name = type(stage).__name__
        if name == "FilePartitioningStage":
            # ManifestReader's framework partitioner only discovers input paths
            # and is not part of the audio AgentReady contract surface.
            continue
        try:
            contract = foundation.build_contract(stage)
        except Exception as exc:  # noqa: BLE001 - inability to inspect must fail closed
            issues.append(
                f"{name}: disk-write contract could not be inspected "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        if not contract.gates.writes_to_disk:
            continue
        # The stage declares its own output params, so a new writer needs no edit here.
        # ``None`` still means NOT DECLARED and still fails closed: a stage that says it
        # writes to disk without naming where cannot be proven isolated, and guessing from
        # parameter names would risk a smoke writing into the caller's real output tree.
        adapter = contract.gates.output_path_params
        if adapter is None:
            issues.append(
                f"{name}: declares writes_to_disk=True but does not declare "
                "output_path_params, so its output cannot be redirected into smoke storage"
            )
            continue
        if name == "CreateInitialManifestReadSpeechStage":
            if getattr(stage, "auto_download", True):
                issues.append(
                    f"{name}: auto_download must be disabled for a pre-staged smoke"
                )
            continue
        if not adapter:
            # ``[]`` is a positive claim -- "I write to disk through no redirectable
            # parameter" -- and for the one stage above that is both true and safe, which is
            # why it is handled by name just before this. Accepted from ANY stage, though, it
            # became the easiest way past the whole check: a writer with a hardcoded or
            # derived destination declares an empty list, the loop below has nothing to
            # iterate, and the smoke is pronounced isolated while the stage writes into the
            # caller's real output tree. Same reasoning as the undeclared case -- unprovable
            # isolation fails closed -- so the exemption is a name, not a shape.
            issues.append(
                f"{name}: declares writes_to_disk=True with an empty output_path_params, so "
                "there is no parameter to redirect and its isolation cannot be proven"
            )
            continue
        for param in adapter:
            value = getattr(stage, param, None)
            if not isinstance(value, str) or not _inside_smoke_root(
                value,
                output_root,
            ):
                issues.append(
                    f"{name}.{param}: output is not inside temporary smoke storage"
                )
    return issues


def _walk_smoke_stages(stages: list[Any]):  # noqa: ANN202 - private generator
    """Yield configured stages and the concrete children of composites."""
    from nemo_curator.stages.base import CompositeStage

    for stage in stages:
        yield stage
        if isinstance(stage, CompositeStage):
            yield from _walk_smoke_stages(list(stage.decompose()))


def _inside_smoke_root(path: str, root: str) -> bool:
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    return resolved_path == resolved_root or resolved_path.startswith(
        resolved_root + os.sep
    )


def _write_bounded_manifest(
    paths: tuple[str, ...],
    n: int,
) -> tuple[str, int, str]:
    """Concatenate at most ``n`` valid JSON-object rows from ordered local files."""
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="audio_agent_smoke_")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for path in paths:
                try:
                    with open(path, encoding="utf-8") as src:
                        for lineno, line in enumerate(src, 1):
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError as exc:
                                return _bounded_write_error(
                                    tmp,
                                    count,
                                    f"{path}:{lineno} contains invalid JSON: {exc}",
                                )
                            if not isinstance(row, dict):
                                return _bounded_write_error(
                                    tmp,
                                    count,
                                    f"{path}:{lineno} must contain a JSON object",
                                )
                            out.write(line if line.endswith("\n") else line + "\n")
                            count += 1
                            if count >= n:
                                return tmp, count, ""
                except (OSError, UnicodeError) as exc:
                    return _bounded_write_error(
                        tmp,
                        count,
                        f"{path} could not be read: {type(exc).__name__}: {exc}",
                    )
    except OSError as exc:
        return _bounded_write_error(
            tmp,
            count,
            f"bounded manifest could not be written: {type(exc).__name__}: {exc}",
        )
    return tmp, count, ""


def _write_bounded_fleurs_manifest(
    source: Any,  # noqa: ANN401 - copied StageRef
    binding: Any,  # noqa: ANN401 - DatasetBinding
    n: int,
) -> tuple[str, int, str]:
    """Materialize the first ``n`` rows exactly as a pre-staged FLEURS source does."""
    identity_files = list(getattr(binding, "profile_kwargs", {}).get("identity_files") or [])
    if len(identity_files) != 1:
        return "", 0, "pre-staged FLEURS transcript was not uniquely resolved"
    transcript = identity_files[0]
    audio_root = str(binding.profile_source)
    filepath_key = source.params.get("filepath_key", "audio_filepath")
    text_key = source.params.get("text_key", "text")
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="audio_agent_smoke_fleurs_")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out, open(
            transcript,
            encoding="utf-8",
        ) as src:
            for line in src:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                row = {
                    filepath_key: os.path.abspath(os.path.join(audio_root, parts[1])),
                    text_key: parts[2],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                if count >= n:
                    break
    except (OSError, UnicodeError) as exc:
        return _bounded_write_error(
            tmp,
            count,
            f"{transcript} could not be read: {type(exc).__name__}: {exc}",
        )
    return tmp, count, ""


def _bounded_write_error(tmp: str, count: int, error: str) -> tuple[str, int, str]:
    with contextlib.suppress(OSError):
        os.remove(tmp)
    return "", count, error


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)


def _estimate(data_profile: dict[str, Any] | None) -> dict[str, Any]:
    dp = data_profile or {}
    return {
        "input_count": dp.get("num_files", 0),
        "total_duration_sec": dp.get("total_duration_sec", 0.0),
        "note": "precise $/GPU-hour costing is deferred; time estimate comes from a prior smoke run",
    }


def _recipe_outputs(recipe: Recipe, output_dir: str | None) -> list[str]:
    """Every location the recipe writes to, in stage order.

    Scans the shared ``OUTPUT_LOCATION_PARAMS`` set, so intermediates a stage produces as a
    side effect (resampled dirs, per-speaker audio, RTTM dirs) are recorded rather than being
    invisible. ``path`` is kept as a legacy alias. ``raw_data_dir`` is a source
    cache/input, not a curated output. The verb-level ``output_dir`` argument is
    retained for API compatibility but intentionally never invents an output the
    executable recipe did not write.
    """
    from nemo_curator.audio_agent.recipe import OUTPUT_LOCATION_PARAMS

    keys = [*sorted(OUTPUT_LOCATION_PARAMS - {"raw_data_dir"}), "path"]
    outs: list[str] = []
    for s in recipe.stages:
        for key in keys:
            v = s.params.get(key)
            if isinstance(v, str) and v not in outs:
                outs.append(v)
    _ = output_dir  # legacy no-op; stage params are execution truth
    return outs


def _stage_metrics(results: list[Any] | None) -> dict[str, Any]:
    from nemo_curator.audio_agent.report import _dedup_stage_perf

    return _dedup_stage_perf(results or [])


def _result_rows(results: list[Any] | None) -> Iterator[dict[str, Any]]:
    """Yield logical rows from one-row tasks and batched document carriers.

    AudioTask-like results carry one mapping in ``task.data``. ``DocumentBatch``
    carries a pandas/Arrow table and exposes ``to_pandas``. Keeping this adapter
    structural avoids importing either dataframe dependency in the agent core
    while ensuring evidence counts and previews describe the same logical rows.
    """
    for task in results or []:
        data = getattr(task, "data", None)
        if isinstance(data, dict):
            yield data
            continue
        to_pandas = getattr(task, "to_pandas", None)
        if not callable(to_pandas):
            continue
        frame = to_pandas()
        to_dict = getattr(frame, "to_dict", None)
        if not callable(to_dict):
            continue
        records = to_dict(orient="records")
        if not isinstance(records, list):
            continue
        for row in records:
            if isinstance(row, dict):
                yield row


def _examples_from_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in row.items() if _jsonable(v)}
        for row in rows[:limit]
    ]


def _examples(results: list[Any] | None, *, limit: int) -> list[dict[str, Any]]:
    """Return at most ``limit`` JSON-safe logical rows from result carriers."""
    rows: list[dict[str, Any]] = []
    for row in _result_rows(results):
        rows.append(row)
        if len(rows) >= limit:
            break
    return _examples_from_rows(rows, limit=limit)


def _is_value_empty(v: Any) -> bool:  # noqa: ANN401
    """A produced value that carries no content (None / blank string / empty container)."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict, tuple)):
        return len(v) == 0
    # DataFrame rows use NaN for a column absent from one record. Treat that as
    # missing evidence without importing pandas/numpy into this module.
    try:
        return bool(v != v)
    except (TypeError, ValueError):
        return False


def _empty_required_outputs(rec: Recipe, rows: list[dict[str, Any]]) -> list[str]:
    """Required output keys (from the recipe's ``output_completeness`` criteria) that are
    missing or empty in at least one bounded retained row.

    ``output_completeness`` defaults to per-retained-item semantics, so one
    missing/blank value is enough to withhold the smoke token. If retained rows
    cannot be inspected at all, a deterministic required output is unverifiable
    and therefore also cannot authorize a full run.
    """
    from nemo_curator.audio_agent.acceptance import parse_criteria

    keys: list[str] = []
    for c in parse_criteria(getattr(rec, "acceptance_criteria", None) or []):
        if c.type == "output_completeness" and c.is_deterministic:
            key = c.field_name or (c.compiles_to if c.compiles_to and c.compiles_to != "producible_role" else None)
            if key:
                keys.append(key)
    empty: list[str] = []
    for key in keys:
        if not rows or any(key not in row or _is_value_empty(row.get(key)) for row in rows):
            empty.append(key)
    return empty


def _jsonable(v: Any) -> bool:  # noqa: ANN401
    """Preview-safe if a scalar, or a JSON-serializable nested dict/list/tuple.

    Scalars pass directly; nested containers (e.g. ``metrics`` with SQUIM or
    WER/CER scores, or ``segments``) are kept when they round-trip through
    ``json.dumps`` (``default=str`` matches the CLI/MCP emitter and tolerates a
    stray numpy scalar). Bare non-JSON objects (waveform tensors/arrays) are
    dropped. The report is still ``_safety.redact``-ed and capped by ``limit``.
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return True
    if isinstance(v, (dict, list, tuple)):
        try:
            json.dumps(v, default=str)
        except (TypeError, ValueError, RecursionError):
            return False
        return True
    return False


def _locate_manifest_files(
    output: str,
) -> tuple[Any | None, list[str], str]:
    """Resolve a local/cloud manifest location through one fsspec filesystem.

    Paths returned here are filesystem-relative and must be opened through the
    returned ``fs`` object. A directly named file keeps the historical behavior
    of being scannable regardless of suffix; directories recursively include
    only ``.json``/``.jsonl`` files.
    """
    try:
        from fsspec.core import url_to_fs

        fs, resolved = url_to_fs(os.path.expanduser(output))
        if fs.isfile(resolved):
            return fs, [resolved], "ok"
        if fs.isdir(resolved):
            found = sorted(
                str(path)
                for path in fs.find(
                    resolved,
                    withdirs=False,
                    detail=False,
                )
                if str(path).endswith((".jsonl", ".json"))
            )
            return fs, found, "ok" if found else "no_manifest"
        return fs, [], "missing" if not fs.exists(resolved) else "no_manifest"
    except Exception:  # noqa: BLE001 - evidence readback must not fail a completed run
        return None, [], "unreadable"


def _scan_output_inventory(output: str) -> dict[str, Any]:
    """Inventory a non-manifest file/directory without claiming row evidence."""
    summary: dict[str, Any] = {
        "status": "unavailable",
        "output": output,
        "files": 0,
        "bytes": 0,
        "suffixes": {},
        "read_errors": 0,
    }
    try:
        from fsspec.core import url_to_fs

        fs, resolved = url_to_fs(os.path.expanduser(output))
        if fs.isfile(resolved):
            files = [resolved]
        elif fs.isdir(resolved):
            files = sorted(
                str(path)
                for path in fs.find(
                    resolved,
                    withdirs=False,
                    detail=False,
                )
            )
        else:
            summary["status"] = "missing"
            return summary
        for path in files:
            try:
                info = fs.info(path)
                summary["bytes"] += int(info.get("size") or 0)
                suffix = os.path.splitext(str(path))[1].lower() or "<none>"
                suffixes = summary["suffixes"]
                suffixes[suffix] = int(suffixes.get(suffix) or 0) + 1
            except Exception:  # noqa: BLE001 - one unreadable entry stays explicit
                summary["read_errors"] += 1
        summary["files"] = len(files)
        if summary["read_errors"]:
            summary["status"] = "partial"
        else:
            summary["status"] = "complete" if files else "empty"
    except Exception:  # noqa: BLE001 - backend failures are evidence, not crashes
        summary["status"] = "unreadable"
        summary["read_errors"] = 1
    return summary


def _count_output_rows(output: str) -> int:
    fs, files, _status = _locate_manifest_files(output)
    if fs is None:
        return 0
    total = 0
    for path in files:
        try:
            with fs.open(path, "rt", encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
        except Exception:  # noqa: BLE001 - output inventory is best-effort
            continue
    return total


def _scan_terminal_output(  # noqa: C901, PLR0912, PLR0915 - streaming evidence states
    outputs: list[str],
    *,
    limit: int = _EVIDENCE_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan the terminal manifest completely while retaining a bounded row preview.

    The last declared output is authoritative. A missing/empty terminal product
    must never borrow valid rows from an earlier intermediate. The summary stores
    counts only (including per-field coverage), so exhaustive checking does not
    retain transcripts or scale memory with the corpus size.
    """
    terminal = str(outputs[-1]) if outputs else ""
    summary: dict[str, Any] = {
        "status": "unavailable",
        "terminal_output": terminal,
        "files": 0,
        "readable_files": 0,
        "valid_rows": 0,
        "malformed_rows": 0,
        "blank_rows": 0,
        "read_errors": 0,
        "field_scope": "top_level",
        "fields": {},
    }
    if not terminal:
        return [], summary

    fs, files, locate_status = _locate_manifest_files(terminal)
    summary["files"] = len(files)
    if locate_status != "ok" or fs is None:
        summary["status"] = locate_status
        if locate_status == "unreadable":
            summary["read_errors"] = 1
        return [], summary

    preview: list[dict[str, Any]] = []
    fields: dict[str, dict[str, Any]] = summary["fields"]
    for path in files:
        try:
            fh = fs.open(path, "rt", encoding="utf-8")
        except Exception:  # noqa: BLE001 - backend errors become explicit evidence state
            summary["read_errors"] += 1
            continue
        summary["readable_files"] += 1
        try:
            with fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        summary["blank_rows"] += 1
                        continue
                    try:
                        obj = json.loads(line)
                    except (TypeError, ValueError):
                        summary["malformed_rows"] += 1
                        continue
                    if not isinstance(obj, dict):
                        summary["malformed_rows"] += 1
                        continue

                    summary["valid_rows"] += 1
                    if len(preview) < limit:
                        preview.append(obj)
                    for key, value in obj.items():
                        stat = fields.setdefault(
                            str(key),
                            {
                                "present": 0,
                                "non_empty": 0,
                                "numeric": 0,
                            },
                        )
                        stat["present"] += 1
                        if not _is_value_empty(value):
                            stat["non_empty"] += 1
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                        ):
                            number = float(value)
                            stat["numeric"] += 1
                            stat["sum"] = stat.get("sum", 0.0) + number
                            stat["min"] = min(number, stat.get("min", number))
                            stat["max"] = max(number, stat.get("max", number))
        except Exception:  # noqa: BLE001 - remote iteration can raise backend-specific errors
            summary["read_errors"] += 1

    if summary["read_errors"] and not summary["readable_files"]:
        summary["status"] = "unreadable"
    elif summary["read_errors"] or summary["malformed_rows"] or summary["blank_rows"]:
        summary["status"] = "partial"
    elif not summary["valid_rows"]:
        summary["status"] = "empty"
    else:
        summary["status"] = "complete"
    rows = int(summary["valid_rows"])
    for stat in fields.values():
        if rows > 0 and int(stat.get("numeric") or 0) == rows:
            stat["mean"] = float(stat.get("sum") or 0.0) / rows
    return preview, summary


def _aggregate_metrics(output_scan: dict[str, Any]) -> dict[str, float]:
    """Return complete-corpus means for wholly numeric terminal fields.

    Aggregate acceptance is evidence-backed only when every serialized row was
    read successfully and every row carries a finite number for that field.
    Partial scans and mixed/missing values stay unverifiable.
    """
    if str(output_scan.get("status") or "") != "complete":
        return {}
    rows = int(output_scan.get("valid_rows") or 0)
    if rows <= 0:
        return {}
    metrics: dict[str, float] = {}
    for field, stat in dict(output_scan.get("fields") or {}).items():
        if (
            int(stat.get("present") or 0) == rows
            and int(stat.get("numeric") or 0) == rows
            and isinstance(stat.get("mean"), (int, float))
            and not isinstance(stat.get("mean"), bool)
            and math.isfinite(float(stat["mean"]))
        ):
            metrics[str(field)] = float(stat["mean"])
    return metrics
