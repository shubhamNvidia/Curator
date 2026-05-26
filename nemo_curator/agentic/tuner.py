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
"""Resource + executor tuner.

The tuner is the *minimal* layer between stage selection and the executor.
It runs after :func:`select_stages` and before the validator. Given:

- The user-supplied ``ClusterProfile`` (CPU count, GPU count, GPU mem).
- The selected list of stage references (``StageRef``).
- An optional ``forced_execution_mode``.

it emits, on each ``StageRef``:

- ``resources``: a concrete ``ResourceSpec`` with the per-worker
  ``cpus`` and ``gpus`` fractions. The GPU fraction comes from VRAM
  packing because Ray must know the per-actor GPU ask at placement time
  and there is no way for the backend to infer it from the stage code.
- ``batch_size``: when the stage's card overrides the default.
- ``backend_hints``: intentionally ``None``. ``num_workers`` and
  ``slots_per_actor`` are left to Xenna's autoscaler — it's the
  authoritative source on actor count vs backlog/throughput, and any
  number the tuner pinned would become stale the moment the workload
  drifts.
- ``tuner_reasons``: human-readable explanation chips.

and an ``ExecutorConfig`` for the top-level IR. The tuner is
deterministic — there is no LLM call here.

Why the tuner exists
====================

Stage cards declare a *default* resource ask, but the cards are often
wrong (or absent) for the agentic flow:

- ``InferenceAsrNemoStage`` historically declared ``gpus=0`` even though
  Parakeet-1.1B on CPU is unusable in a streaming pipeline.
- ``InferenceSortformerStage`` declares ``gpu_memory_gb=8`` (an absolute
  ask), which the runner would otherwise map to a fractional ``gpus``
  based on the *cluster's* GPU memory rather than the card's idea of it.
- ``UTMOSFilterStage`` / ``SIGMOSFilterStage`` declare ``gpus=0.5`` —
  fine for streaming, wasteful in batch mode.

``STAGE_RESOURCE_PROFILES`` is a small truth table that classifies every
stage by its real GPU appetite (``full`` / ``frac`` / ``cpu``). VRAM
packing converts that classification + the cluster's per-GPU memory into
a concrete ``gpus_per_worker``. Everything else (how many workers, when
to scale up/down, per-actor concurrency) is the backend's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nemo_curator.agentic.cards import ResourceSpec, StageCard
from nemo_curator.agentic.ir import (
    ClusterProfile,
    ExecutorConfig,
    StageRef,
)
from nemo_curator.agentic.registry import CapabilityRegistry


# ----------------------------------------------------------------------------
# GPU appetite truth table
# ----------------------------------------------------------------------------


GpuClass = Literal["full", "frac", "cpu"]


@dataclass(frozen=True)
class StageResourceProfile:
    """The tuner's own opinion of a stage's resource needs.

    Independent of what the stage card declares. Card defaults can be
    misleading (a GPU-heavy stage might claim ``gpus=0``); this table is
    what the tuner uses to allocate.

    Fields:

    - ``gpu_class``: ``full`` = needs a whole GPU per worker; ``frac`` =
      can share a GPU with at least one other worker; ``cpu`` = pure
      CPU stage, never claims GPU.
    - ``gpu_memory_gb``: minimum VRAM in GB. Used to convert from a
      fractional-GPU ask to an absolute ask when the cluster's per-GPU
      memory is known.
    - ``cpus_per_worker``: CPU cores per worker (mostly 1.0).
    - ``io_bound``: true for stages whose work is dominated by disk or
      subprocess IO (ffmpeg, file IO). The tuner sets ``slots_per_actor``
      > 1 on these to keep the CPU fed.
    - ``supports_batch``: stage overrides ``process_batch`` — the tuner
      may set a non-default ``batch_size``.
    - ``min_workers`` / ``max_workers``: hard floor / ceiling that the
      streaming and batch allocators clamp to.
    """

    gpu_class: GpuClass
    gpu_memory_gb: float = 0.0
    cpus_per_worker: float = 1.0
    io_bound: bool = False
    supports_batch: bool = False
    min_workers: int = 1
    max_workers: int | None = None
    notes: str = ""


# Truth table. Add a row whenever a new stage is introduced.
# Stages absent from this map are treated as cheap CPU (cpus_per_worker=1.0,
# no GPU, not io-bound). That is the safe default.
STAGE_RESOURCE_PROFILES: dict[str, StageResourceProfile] = {
    # --- GPU-heavy: needs a whole GPU per worker ---------------------------
    "InferenceAsrNemoStage": StageResourceProfile(
        gpu_class="full", gpu_memory_gb=8.0, supports_batch=True,
        notes="Parakeet-1.1B; CPU inference is unusable.",
    ),
    "NeMoASRAlignerStage": StageResourceProfile(
        gpu_class="full", gpu_memory_gb=8.0, supports_batch=True,
        notes="Forced-alignment NeMo model.",
    ),
    "SplitASRAlignJoinStage": StageResourceProfile(
        gpu_class="full", gpu_memory_gb=10.0, supports_batch=True,
        notes="Long-form ASR+align with split/join.",
    ),
    "SpeakerSeparationStage": StageResourceProfile(
        gpu_class="full", gpu_memory_gb=8.0,
        notes="Voicebox/SepFormer-style fan-out separation.",
    ),
    "PyAnnoteDiarizationStage": StageResourceProfile(
        gpu_class="full", gpu_memory_gb=4.0,
        notes="pyannote pipeline; GPU recommended.",
    ),
    "AudioDataFilterStage": StageResourceProfile(
        gpu_class="full", gpu_memory_gb=8.0,
        notes="Composite filter that internally runs MOS + diarization.",
    ),
    "WhisperXVADStage": StageResourceProfile(
        gpu_class="full", gpu_memory_gb=4.0,
        notes="Whisper-X VAD; GPU only.",
    ),
    # NB: WhisperXAsrStage is referenced in the clarifier
    # (``text.asr_backend == 'whisper'``) but is not yet registered as a
    # real stage card. The selector falls back to ``InferenceAsrNemoStage``
    # when that branch is taken, so we don't need a resource profile here.
    # Re-add this entry once ``nemo_curator/stages/audio/_cards/
    # WhisperXAsrStage/stage_card.yaml`` (and its implementation) land.
    # --- GPU-shareable: can run multiple workers per GPU --------------------
    "InferenceSortformerStage": StageResourceProfile(
        gpu_class="frac", gpu_memory_gb=8.0,
        notes="Sortformer diarizer; declares gpu_memory_gb=8 in its card.",
    ),
    "UTMOSFilterStage": StageResourceProfile(
        gpu_class="frac", gpu_memory_gb=2.0,
        notes="UTMOS22-strong MOS estimator.",
    ),
    "SIGMOSFilterStage": StageResourceProfile(
        gpu_class="frac", gpu_memory_gb=2.0,
        notes="Microsoft SIG-MOS ONNX.",
    ),
    # --- CPU stages, IO-bound (ffmpeg / subprocess) -------------------------
    "ResampleAudioStage": StageResourceProfile(
        gpu_class="cpu", io_bound=True,
        notes="Shells out to ffmpeg; benefits from slots_per_actor > 1.",
    ),
    "MonoConversionStage": StageResourceProfile(
        gpu_class="cpu", io_bound=True,
        notes="Reads + decodes audio into memory.",
    ),
    "SegmentExtractionStage": StageResourceProfile(
        gpu_class="cpu", io_bound=True, supports_batch=True,
        notes="ffmpeg trim/encode for per-clip output.",
    ),
    "VADSegmentationStage": StageResourceProfile(
        gpu_class="cpu", cpus_per_worker=1.0,
        notes="Silero VAD; tiny enough to run on CPU.",
    ),
    "BandFilterStage": StageResourceProfile(
        gpu_class="cpu", cpus_per_worker=4.0,
        notes="Spectral analysis; declares cpus=4 in its card.",
    ),
    # --- Cheap CPU stages: no overrides needed ------------------------------
    "ManifestReader": StageResourceProfile(gpu_class="cpu"),
    "ManifestReaderStage": StageResourceProfile(gpu_class="cpu"),
    "ManifestWriterStage": StageResourceProfile(gpu_class="cpu"),
    "FilePartitioningStage": StageResourceProfile(gpu_class="cpu"),
    "GetAudioDurationStage": StageResourceProfile(gpu_class="cpu"),
    "TimestampMapperStage": StageResourceProfile(gpu_class="cpu"),
    "MergeAlignmentDiarizationStage": StageResourceProfile(gpu_class="cpu"),
    "PreserveByValueStage": StageResourceProfile(gpu_class="cpu", supports_batch=True),
    "GetPairwiseWerStage": StageResourceProfile(gpu_class="cpu"),
    "AudioToDocumentStage": StageResourceProfile(gpu_class="cpu", supports_batch=True),
    "JoinSplitAudioMetadataStage": StageResourceProfile(gpu_class="cpu"),
    "SegmentConcatenationStage": StageResourceProfile(gpu_class="cpu"),
    "SplitLongAudioStage": StageResourceProfile(gpu_class="cpu"),
    "ALMDataBuilderStage": StageResourceProfile(gpu_class="cpu"),
    "ALMDataOverlapStage": StageResourceProfile(gpu_class="cpu"),
    "CreateInitialManifestFleursStage": StageResourceProfile(gpu_class="cpu"),
    "CreateInitialManifestReadSpeechStage": StageResourceProfile(gpu_class="cpu"),
}


def _profile_for(stage_name: str) -> StageResourceProfile:
    """Return the profile for ``stage_name``, defaulting to cheap-CPU."""

    return STAGE_RESOURCE_PROFILES.get(
        stage_name,
        StageResourceProfile(gpu_class="cpu"),
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


@dataclass
class TunerInputs:
    """Bundle of inputs the tuner needs.

    ``execution_mode`` is intentionally **not** a user input — the tuner
    picks ``streaming`` vs ``batch`` based on whether the pipeline's
    minimum VRAM-packed demand fits the cluster. Callers wanting to
    override may set ``forced_execution_mode``; this is reserved for
    tests and migrations and not exposed to the web UI.
    """

    stages: list[StageRef]
    registry: CapabilityRegistry
    cluster: ClusterProfile
    forced_execution_mode: Literal["streaming", "batch"] | None = None
    backend: Literal["xenna", "ray_actor_pool", "ray_data"] | None = None
    expected_input_files: int | None = None


@dataclass
class TunerResult:
    """What :func:`tune` returns."""

    stages: list[StageRef]
    executor_config: ExecutorConfig
    reasons: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Heaviness + VRAM packing constants
# ----------------------------------------------------------------------------


# Hard cap on workers-per-GPU. Beyond ~8, Ray scheduling overhead and CUDA
# stream contention dominate any throughput win.
MAX_WORKERS_PER_GPU: int = 8

# 30% headroom for activations / KV cache / CUDA context / Python overhead
# on top of the raw weight VRAM.
VRAM_OVERHEAD: float = 1.30


# Fallback weights when a stage card has no ``params_total``. Keeps the
# allocator working on freshly added cards before the inspector has run.
_DEFAULT_WEIGHT_BY_CLASS: dict[GpuClass, int] = {
    "full": 5,
    "frac": 2,
    "cpu": 1,
}


# ----------------------------------------------------------------------------
# Tune entry point
# ----------------------------------------------------------------------------


def tune(inputs: TunerInputs) -> TunerResult:
    """Allocate per-worker resource fractions for every stage and pick an
    executor config. The backend (Xenna/Ray) handles ``num_workers`` and
    ``slots_per_actor`` autoscaling on its own.

    Order of operations:

    1. Compute each GPU stage's ``gpus_per_worker`` from VRAM packing
       (the *fraction* decision). This is the only number the tuner is
       authoritative about — the backend can't infer VRAM by itself.
    2. Compute ``cost_weight`` from card-derived ``params_total`` purely
       as informational ``tuner_reasons`` chips.
    3. Decide ``execution_mode`` automatically: if every GPU stage can
       place one worker concurrently (``Σ gpus_per_worker ≤ cluster.gpus``)
       streaming is feasible; otherwise we fall back to batch so stages
       run sequentially and each gets the whole GPU pool during its turn.
    4. Emit ``resources.cpus`` / ``resources.gpus`` only and leave
       ``backend_hints`` unset so Xenna's autoscaler is free to size the
       actor pool (and ``slots_per_actor``) on its own.
    """

    # ---- Pre-pass: per-stage fractions + informational weights ---------------
    fractions: dict[str, float] = {}
    weights: dict[str, int] = {}
    derive_reasons: list[str] = []
    for ref in inputs.stages:
        card = _card_for(inputs.registry, ref.stage)
        profile = _profile_for(ref.stage)
        gpw, why_gpw = _gpus_per_worker_from_vram(
            card=card, profile=profile, cluster=inputs.cluster,
        )
        cw, why_cw = _cost_weight_from_card(card=card, profile=profile)
        fractions[ref.stage] = gpw
        weights[ref.stage] = cw
        derive_reasons.append(f"{ref.stage}: {why_gpw}; cost_weight={cw} ({why_cw})")

    # ---- Auto-decide execution mode ------------------------------------------
    execution_mode, mode_reason = _decide_execution_mode(
        stages=inputs.stages,
        cluster=inputs.cluster,
        fractions=fractions,
        forced=inputs.forced_execution_mode,
    )
    backend = inputs.backend or _pick_backend(execution_mode)

    # ---- Per-stage allocation -------------------------------------------------
    tuned: list[StageRef] = []
    reasons: list[str] = []
    for ref in inputs.stages:
        card = _card_for(inputs.registry, ref.stage)
        profile = _profile_for(ref.stage)
        new_ref, why = _tune_one(
            ref,
            card=card,
            profile=profile,
            execution_mode=execution_mode,
            gpus_per_worker=fractions[ref.stage],
        )
        tuned.append(new_ref)
        reasons.extend(f"{ref.stage}: {w}" for w in why)

    config = _build_executor_config(
        backend=backend,
        execution_mode=execution_mode,
        cluster=inputs.cluster,
        mode_reason=mode_reason,
    )
    return TunerResult(stages=tuned, executor_config=config, reasons=reasons + derive_reasons)


# ----------------------------------------------------------------------------
# Fraction (gpus_per_worker) — VRAM packing
# ----------------------------------------------------------------------------


def _gpus_per_worker_from_vram(
    *,
    card: StageCard | None,
    profile: StageResourceProfile,
    cluster: ClusterProfile,
) -> tuple[float, str]:
    """Pack workers onto a GPU based on the model's VRAM footprint.

    Formula::

        vram_needed = est_vram_inference_gb × VRAM_OVERHEAD
        pack        = clamp(floor(gpu_memory_gb / vram_needed), 1, MAX_WORKERS_PER_GPU)
        gpus_per_worker = 1 / pack

    Fallback: when the stage card lacks ``est_vram_inference_gb`` (no
    inspection done yet), use the gpu_class default (1.0 for ``full``,
    0.5 for ``frac``). Returns ``0.0`` for CPU stages or when the
    cluster has no GPUs.
    """

    if profile.gpu_class == "cpu" or cluster.gpus <= 0:
        return 0.0, "cpu stage or no GPUs on cluster"

    vram = 0.0
    if card is not None and card.models:
        for m in card.models:
            if m.est_vram_inference_gb and m.est_vram_inference_gb > vram:
                vram = float(m.est_vram_inference_gb)

    if vram <= 0 or cluster.gpu_memory_gb <= 0:
        default = 1.0 if profile.gpu_class == "full" else 0.5
        return default, (
            f"vram unknown for stage; falling back to gpu_class default ({default})"
        )

    need = vram * VRAM_OVERHEAD
    if need > cluster.gpu_memory_gb:
        # Model doesn't fit on a single card. We still need a number to
        # avoid crashing the planner; ask for the whole GPU and let the
        # audit pass warn. (A future iteration could escalate to a tensor-
        # parallel path; out of scope for v1.)
        return 1.0, (
            f"model needs {need:.1f} GB but cluster has {cluster.gpu_memory_gb:.0f} "
            "GB per GPU; asking for 1.0 (may OOM at runtime)"
        )

    pack = max(1, min(MAX_WORKERS_PER_GPU, int(cluster.gpu_memory_gb // need)))
    return round(1.0 / pack, 4), (
        f"pack {pack} workers/GPU ({vram:.2f} GB × {VRAM_OVERHEAD:.2f} = "
        f"{need:.2f} GB fit in {cluster.gpu_memory_gb:.0f} GB)"
    )


# ----------------------------------------------------------------------------
# Cost weight — heaviness from params
# ----------------------------------------------------------------------------


def _cost_weight_from_card(
    *, card: StageCard | None, profile: StageResourceProfile,
) -> tuple[int, str]:
    """Map ``params_total`` to a heaviness weight in ``[1, 10]``.

    Piecewise closed form roughly:

    - <= 10M params  → weight 1
    - 100M params    → weight 3
    - 1B params      → weight 7
    - 10B+ params    → weight 10 (cap)

    Fallback when no card or no params: gpu_class default (5/2/1).
    """

    if card is None or not card.models:
        return _DEFAULT_WEIGHT_BY_CLASS[profile.gpu_class], "no card/models on stage"

    biggest = 0
    for m in card.models:
        if m.params_total and m.params_total > biggest:
            biggest = int(m.params_total)
    if biggest <= 0:
        return _DEFAULT_WEIGHT_BY_CLASS[profile.gpu_class], "no params_total on models"

    import math
    # log10 spacing: 1e7 → 1, 1e8 → 3, 1e9 → 7, 1e10 → 10
    raw = math.log10(max(biggest, 1e6) / 1e7) * 2 + 1
    weight = max(1, min(10, int(round(raw))))
    return weight, f"params={biggest / 1e6:.1f}M"


# ----------------------------------------------------------------------------
# Execution mode decision
# ----------------------------------------------------------------------------


def _decide_execution_mode(
    *,
    stages: list[StageRef],
    cluster: ClusterProfile,
    fractions: dict[str, float],
    forced: Literal["streaming", "batch"] | None,
) -> tuple[Literal["streaming", "batch"], str]:
    """Pick streaming or batch automatically.

    Streaming is viable only when every GPU stage can spin up at least
    one worker concurrently::

        sum(gpus_per_worker_i for GPU stages)  ≤  cluster.gpus

    Otherwise we fall back to batch so stages can run sequentially and
    each gets the whole GPU pool during its turn.

    If ``forced`` is set we still compute the auto-decision so we can
    accurately label the result as either *preserved-on-re-tune* (when
    forced == auto) or genuinely overridden (forced != auto).
    """

    # ---- Auto-decision (always computed, even if a forced mode wins) ----
    if cluster.gpus <= 0:
        auto: Literal["streaming", "batch"] = "batch"
        auto_reason = "cluster has 0 GPUs → batch"
    else:
        floor = 0.0
        for ref in stages:
            if _profile_for(ref.stage).gpu_class == "cpu":
                continue
            floor += fractions.get(ref.stage, 0.0)
        if floor <= float(cluster.gpus) + 1e-6:
            auto = "streaming"
            auto_reason = (
                f"streaming feasible (Σ min gpus_per_worker = {floor:.2f} ≤ "
                f"cluster.gpus = {cluster.gpus})"
            )
        else:
            auto = "batch"
            auto_reason = (
                f"streaming infeasible (Σ min gpus_per_worker = {floor:.2f} > "
                f"cluster.gpus = {cluster.gpus}) → auto-batch"
            )

    if forced is None:
        return auto, auto_reason
    if forced == auto:
        return forced, f"{auto_reason} (preserved from first-pass auto-decision)"
    return forced, (
        f"forced by caller (forced={forced}, auto-decision would have been "
        f"{auto}: {auto_reason})"
    )


# ----------------------------------------------------------------------------
# Per-stage tuner
# ----------------------------------------------------------------------------


def _tune_one(
    ref: StageRef,
    *,
    card: StageCard | None,
    profile: StageResourceProfile,
    execution_mode: Literal["streaming", "batch"],
    gpus_per_worker: float,
) -> tuple[StageRef, list[str]]:
    """Compute resource fractions for a single stage.

    ``backend_hints`` is intentionally left ``None`` so the backend's
    autoscaler is in charge of ``num_workers`` and ``slots_per_actor``.
    The tuner only emits the per-worker ``cpus`` / ``gpus`` fractions
    (which Ray must know at actor-placement time) and an optional
    ``batch_size`` override.
    """

    reasons: list[str] = list(ref.tuner_reasons)
    cpus_per_worker = profile.cpus_per_worker
    if card is not None and card.resources.cpus > cpus_per_worker:
        cpus_per_worker = card.resources.cpus

    resources = ResourceSpec(cpus=cpus_per_worker, gpus=round(gpus_per_worker, 4))
    batch_size = _pick_batch_size(card, profile, ref, execution_mode)

    update: dict = {
        "resources": resources,
        "backend_hints": None,
        "tuner_reasons": reasons,
    }
    if batch_size is not None:
        update["batch_size"] = batch_size
        reasons.append(f"batch_size={batch_size}")

    return ref.model_copy(update=update), reasons


# ----------------------------------------------------------------------------
# Worker count + batch_size + slots
# ----------------------------------------------------------------------------


def _cpu_alloc_pct_for(cluster: ClusterProfile) -> float:
    """The fraction of cluster.cpus actually offered to Xenna for actor placement.

    Mirrors the value emitted by ``_build_executor_config``: the default
    Xenna pipeline reserves 5% of CPUs for the head + raylet, and we drop
    to 0.85 when the caller has already pre-reserved CPUs. The tuner MUST
    use this same fraction when sizing num_workers, otherwise we over-
    promise CPU and Xenna will throttle actor placement at runtime.
    """

    return 0.85 if cluster.reserved_cpus > 0 else 0.95


def _pick_batch_size(
    card: StageCard | None,
    profile: StageResourceProfile,
    ref: StageRef,
    execution_mode: str,  # noqa: ARG001 — kept for future heuristics
) -> int | None:
    """Override the card's batch_size only when we have a reason to."""

    if ref.batch_size is not None:
        return ref.batch_size
    if card is None:
        return None
    if not card.supports_process_batch or not profile.supports_batch:
        return None
    # Trust the card's default — the tuner doesn't yet have a dataset
    # profile rich enough to second-guess it. We leave the field as
    # ``None`` so the runner uses the dataclass default verbatim.
    return None


# ----------------------------------------------------------------------------
# Top-level executor config + backend pick
# ----------------------------------------------------------------------------


def _pick_backend(
    execution_mode: Literal["streaming", "batch"],
) -> Literal["xenna", "ray_actor_pool"]:
    """Default backend per execution mode.

    The user's ``streaming``/``batch`` choice from the web form maps to a
    backend deterministically:

    - ``streaming`` → ``xenna`` in streaming mode (Cosmos-Xenna's
      autoscaling executor).
    - ``batch`` → ``ray_actor_pool`` (synchronous, one stage at a time —
      simpler resource model that exactly matches the per-stage GPU
      footprints the tuner produces).
    """

    return "xenna" if execution_mode == "streaming" else "ray_actor_pool"


def _build_executor_config(
    *,
    backend: Literal["xenna", "ray_actor_pool", "ray_data"],
    execution_mode: Literal["streaming", "batch"],
    cluster: ClusterProfile,
    mode_reason: str | None = None,
) -> ExecutorConfig:
    reasons: list[str] = []
    cpu_alloc = _cpu_alloc_pct_for(cluster)
    if cluster.reserved_cpus > 0:
        reasons.append(
            f"reserved_cpus={cluster.reserved_cpus} → cpu_allocation_percentage=0.85",
        )
    if mode_reason:
        reasons.append(f"execution_mode={execution_mode}: {mode_reason}")
    reasons.append(f"backend={backend} (auto-picked for execution_mode={execution_mode})")
    reasons.append(
        "num_workers/slots_per_actor unset → backend autoscaler controls actor count + "
        "per-actor concurrency (only per-worker cpus/gpus fractions are pinned)"
    )
    return ExecutorConfig(
        backend=backend,
        execution_mode=execution_mode,
        cpu_allocation_percentage=cpu_alloc,
        reserved_cpus=cluster.reserved_cpus,
        reserved_gpus=cluster.reserved_gpus,
        tuner_reasons=reasons,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _card_for(registry: CapabilityRegistry, stage_name: str) -> StageCard | None:
    entry = registry.get(stage_name)
    return entry.card if entry is not None else None


__all__ = [
    "STAGE_RESOURCE_PROFILES",
    "StageResourceProfile",
    "TunerInputs",
    "TunerResult",
    "tune",
]
