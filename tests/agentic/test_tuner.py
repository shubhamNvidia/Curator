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
"""Resource + executor tuner tests.

The tuner is intentionally minimal:

- ``gpus_per_worker`` comes from VRAM packing (model VRAM / cluster
  per-GPU memory) because Ray needs that fraction at placement time.
- ``cpus_per_worker`` is taken from the truth table / card default.
- ``backend_hints`` is left ``None`` — Xenna's autoscaler decides
  ``num_workers`` and ``slots_per_actor`` at runtime.
- ``execution_mode`` is auto-decided from whether one worker per GPU
  stage fits the cluster simultaneously.
"""

from __future__ import annotations

import pytest

from nemo_curator.agentic.cards import ModelRef
from nemo_curator.agentic.deterministic_planner import plan_from_intent
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    OutputFormat,
    Quality,
    Segmentation,
    Speakers,
    TextPolicy,
)
from nemo_curator.agentic.ir import ClusterProfile, StageRef
from nemo_curator.agentic.registry import build_registry
from nemo_curator.agentic.tuner import (
    STAGE_RESOURCE_PROFILES,
    TunerInputs,
    _cost_weight_from_card,
    _gpus_per_worker_from_vram,
    tune,
)


@pytest.fixture(scope="module")
def registry():
    return build_registry(cross_check_runtime=False, eager=False)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _refs(*names: str) -> list[StageRef]:
    return [StageRef(stage=n, params={}) for n in names]


def _tts_intent() -> IntentCategories:
    return IntentCategories(
        output=OutputFormat(sample_rate=24000, channels="mono", resample_input=True),
        segmentation=Segmentation(
            output_unit="single_speaker_clips",
            duration_min_sec=2.0,
            duration_max_sec=30.0,
        ),
        quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5),
        speakers=Speakers(mode=FilterMode.SPLIT),
        text=TextPolicy(transcript_source="generate"),
        raw_prompt="clean TTS clips at 24 kHz",
    )


# ----------------------------------------------------------------------------
# Fraction (gpus_per_worker) tests — VRAM packing
# ----------------------------------------------------------------------------


def test_gpus_per_worker_packs_by_vram(registry) -> None:
    """ASR (~5.6 GB VRAM) on a 24 GB card packs 3 workers per GPU."""

    card = registry.by_name["InferenceAsrNemoStage"].card
    profile = STAGE_RESOURCE_PROFILES["InferenceAsrNemoStage"]

    # 24 GB card → floor(24 / (5.57 * 1.3)) = floor(24/7.24) = 3 packs
    cluster = ClusterProfile(cpus=8, gpus=1, gpu_memory_gb=24.0)
    gpw, why = _gpus_per_worker_from_vram(card=card, profile=profile, cluster=cluster)
    assert gpw == pytest.approx(1.0 / 3, abs=1e-3)
    assert "pack 3" in why

    # 80 GB H100 → floor(80 / 7.24) = 11 → clamped at MAX_WORKERS_PER_GPU=8
    cluster = ClusterProfile(cpus=64, gpus=1, gpu_memory_gb=80.0)
    gpw, why = _gpus_per_worker_from_vram(card=card, profile=profile, cluster=cluster)
    assert gpw == pytest.approx(1.0 / 8, abs=1e-3)
    assert "pack 8" in why

    # 16 GB card → floor(16 / 7.24) = 2
    cluster = ClusterProfile(cpus=8, gpus=1, gpu_memory_gb=16.0)
    gpw, _ = _gpus_per_worker_from_vram(card=card, profile=profile, cluster=cluster)
    assert gpw == pytest.approx(1.0 / 2, abs=1e-3)


def test_gpus_per_worker_falls_back_when_vram_unknown(registry) -> None:
    """No card metadata → gpu_class default (1.0 for full, 0.5 frac)."""

    # All in-tree cards are now populated by the inspector. To exercise
    # the fallback we pass card=None (the canonical "no audit" signal).
    profile_full = STAGE_RESOURCE_PROFILES["InferenceAsrNemoStage"]
    cluster = ClusterProfile(cpus=8, gpus=1, gpu_memory_gb=24.0)
    gpw, why = _gpus_per_worker_from_vram(card=None, profile=profile_full, cluster=cluster)
    assert gpw == 1.0
    assert "fallback" in why or "default" in why

    profile_frac = STAGE_RESOURCE_PROFILES["UTMOSFilterStage"]
    gpw, why = _gpus_per_worker_from_vram(card=None, profile=profile_frac, cluster=cluster)
    assert gpw == 0.5
    assert "fallback" in why or "default" in why


def test_gpus_per_worker_zero_for_cpu_or_no_gpu(registry) -> None:
    card = registry.by_name["ResampleAudioStage"].card
    profile = STAGE_RESOURCE_PROFILES["ResampleAudioStage"]
    cluster = ClusterProfile(cpus=8, gpus=8, gpu_memory_gb=80.0)
    gpw, _ = _gpus_per_worker_from_vram(card=card, profile=profile, cluster=cluster)
    assert gpw == 0.0

    # Zero-GPU cluster — even GPU stages drop to 0.
    asr_card = registry.by_name["InferenceAsrNemoStage"].card
    asr_profile = STAGE_RESOURCE_PROFILES["InferenceAsrNemoStage"]
    zero_cluster = ClusterProfile(cpus=8, gpus=0, gpu_memory_gb=0.0)
    gpw, _ = _gpus_per_worker_from_vram(card=asr_card, profile=asr_profile, cluster=zero_cluster)
    assert gpw == 0.0


# ----------------------------------------------------------------------------
# Cost-weight tests
# ----------------------------------------------------------------------------


def test_cost_weight_from_params(registry) -> None:
    """1.1B Parakeet → heavy weight; ~117M Sortformer → mid; no-params → default."""

    asr = registry.by_name["InferenceAsrNemoStage"].card
    sortformer = registry.by_name["InferenceSortformerStage"].card
    profile_full = STAGE_RESOURCE_PROFILES["InferenceAsrNemoStage"]
    profile_frac = STAGE_RESOURCE_PROFILES["InferenceSortformerStage"]

    w_asr, _ = _cost_weight_from_card(card=asr, profile=profile_full)
    w_sortformer, _ = _cost_weight_from_card(card=sortformer, profile=profile_frac)

    # Parakeet is roughly 10x bigger than Sortformer in params → bigger weight.
    assert w_asr > w_sortformer
    assert 5 <= w_asr <= 10


def test_cost_weight_falls_back_when_params_missing(registry) -> None:
    """No card metadata → gpu_class default weight (1 cpu, 2 frac, 5 full)."""

    profile = STAGE_RESOURCE_PROFILES["UTMOSFilterStage"]
    w, why = _cost_weight_from_card(card=None, profile=profile)
    assert w == 2
    assert "no params" in why.lower() or "no card" in why.lower()

    profile = STAGE_RESOURCE_PROFILES["InferenceAsrNemoStage"]
    w, why = _cost_weight_from_card(card=None, profile=profile)
    assert w == 5  # full default


# ----------------------------------------------------------------------------
# Auto-mode decision tests
# ----------------------------------------------------------------------------


def test_streaming_picked_when_floor_fits(registry) -> None:
    """5-GPU-stage TTS on 2×H100 80 GB → streaming feasible.

    UTMOS / SIGMOS lack inspector data (torch_hub + ONNX, out of HF scope)
    so they fall back to the 0.5 gpu_class default. Floor ≈ 1.375 GPU,
    which fits on 2 GPUs with plenty of headroom.
    """

    cluster = ClusterProfile(cpus=16, gpus=2, gpu_memory_gb=80.0)
    result = tune(TunerInputs(
        stages=_refs(
            "UTMOSFilterStage", "SIGMOSFilterStage",
            "InferenceSortformerStage", "SpeakerSeparationStage",
            "InferenceAsrNemoStage",
        ),
        registry=registry,
        cluster=cluster,
    ))
    assert result.executor_config.execution_mode == "streaming"
    assert result.executor_config.backend == "xenna"


def test_batch_picked_when_floor_exceeds_pool(registry) -> None:
    """Tight 8 GB GPU + 5 GPU stages → ASR needs a whole card, floor > 1."""

    # On an 8 GB card, ASR (5.57 GB × 1.3 = 7.24 GB needed) packs only
    # 1 worker, so gpus_per_worker(ASR)=1.0. The 4 fractional stages
    # contribute ~0.125 each, lifting the floor above 1.0 → auto-batch.
    cluster = ClusterProfile(cpus=8, gpus=1, gpu_memory_gb=8.0)
    result = tune(TunerInputs(
        stages=_refs(
            "UTMOSFilterStage", "SIGMOSFilterStage",
            "InferenceSortformerStage", "SpeakerSeparationStage",
            "InferenceAsrNemoStage",
        ),
        registry=registry,
        cluster=cluster,
    ))
    assert result.executor_config.execution_mode == "batch", (
        f"Expected batch on tight 8 GB GPU; got {result.executor_config.execution_mode}. "
        f"Reasons: {result.executor_config.tuner_reasons}"
    )
    assert result.executor_config.backend == "ray_actor_pool"
    assert any(
        "infeasible" in r and "auto-batch" in r
        for r in result.executor_config.tuner_reasons
    )


def test_batch_picked_when_zero_gpus(registry) -> None:
    cluster = ClusterProfile(cpus=8, gpus=0, gpu_memory_gb=0.0)
    result = tune(TunerInputs(
        stages=_refs("ResampleAudioStage"),
        registry=registry,
        cluster=cluster,
    ))
    assert result.executor_config.execution_mode == "batch"


def test_forced_execution_mode_overrides(registry) -> None:
    """Tests + migrations can pin the mode via forced_execution_mode."""

    cluster = ClusterProfile(cpus=16, gpus=1, gpu_memory_gb=80.0)
    result = tune(TunerInputs(
        stages=_refs("InferenceAsrNemoStage"),
        registry=registry,
        cluster=cluster,
        forced_execution_mode="batch",
    ))
    assert result.executor_config.execution_mode == "batch"
    # When the forced choice differs from auto, the reason must say so.
    reasons = result.executor_config.tuner_reasons
    assert any(
        "forced by caller" in r and "auto-decision would have been streaming" in r
        for r in reasons
    ), f"Expected genuine-override reason; got {reasons}"


def test_forced_mode_matching_auto_says_preserved(registry) -> None:
    """forced_execution_mode == auto-decision → reason calls it 'preserved'."""

    # 8×H100 — streaming is the natural choice for this pipeline.
    cluster = ClusterProfile(cpus=64, gpus=8, gpu_memory_gb=80.0)
    result = tune(TunerInputs(
        stages=_refs(
            "InferenceSortformerStage", "InferenceAsrNemoStage",
        ),
        registry=registry,
        cluster=cluster,
        forced_execution_mode="streaming",
    ))
    assert result.executor_config.execution_mode == "streaming"
    reasons = result.executor_config.tuner_reasons
    assert any(
        "preserved from first-pass auto-decision" in r for r in reasons
    ), f"Expected 'preserved from first-pass auto-decision'; got {reasons}"
    # No false 'forced by caller' phrasing when it actually matches auto.
    assert not any(
        "forced by caller" in r for r in reasons
    ), f"Should NOT say 'forced by caller' when forced == auto; got {reasons}"


# ----------------------------------------------------------------------------
# Weighted fair-share tests
# ----------------------------------------------------------------------------


def test_tuner_does_not_pin_num_workers_or_slots(registry) -> None:
    """The tuner should NOT emit num_workers / slots_per_actor; the
    backend autoscales those on its own. Only the per-worker fractions
    (cpus, gpus) and the optional batch_size are tuner-authoritative.
    """

    cluster = ClusterProfile(cpus=64, gpus=8, gpu_memory_gb=80.0)
    result = tune(TunerInputs(
        stages=_refs(
            "ResampleAudioStage", "MonoConversionStage", "SegmentExtractionStage",
            "UTMOSFilterStage", "InferenceSortformerStage", "InferenceAsrNemoStage",
        ),
        registry=registry,
        cluster=cluster,
    ))
    for ref in result.stages:
        # Either backend_hints is None outright, or all autoscale-related
        # fields are None.
        if ref.backend_hints is not None:
            assert ref.backend_hints.num_workers is None, (
                f"{ref.stage}: tuner must not pin num_workers — backend autoscales"
            )
            assert ref.backend_hints.slots_per_actor is None, (
                f"{ref.stage}: tuner must not pin slots_per_actor — backend autoscales"
            )


def test_tuner_reasons_explain_autoscale_handoff(registry) -> None:
    """Executor-level reasons must say num_workers/slots are autoscaled."""

    cluster = ClusterProfile(cpus=64, gpus=8, gpu_memory_gb=80.0)
    result = tune(TunerInputs(
        stages=_refs("UTMOSFilterStage", "InferenceAsrNemoStage"),
        registry=registry,
        cluster=cluster,
    ))
    reasons = result.executor_config.tuner_reasons
    assert any("backend autoscaler" in r.lower() for r in reasons), (
        f"Expected an explicit autoscaler hand-off chip in tuner_reasons; got: {reasons}"
    )


def test_streaming_per_worker_gpu_floor_fits_cluster(registry) -> None:
    """Σ gpus_per_worker (one actor per GPU stage) must fit cluster.gpus
    when streaming — that is exactly what makes streaming feasible."""

    cluster = ClusterProfile(cpus=64, gpus=8, gpu_memory_gb=80.0)
    result = tune(TunerInputs(
        stages=_refs(
            "UTMOSFilterStage", "SIGMOSFilterStage",
            "InferenceSortformerStage", "SpeakerSeparationStage",
            "InferenceAsrNemoStage",
        ),
        registry=registry,
        cluster=cluster,
    ))
    assert result.executor_config.execution_mode == "streaming", (
        f"Expected streaming on 8 GPUs; got {result.executor_config.execution_mode}. "
        f"Reasons: {result.executor_config.tuner_reasons}"
    )
    floor = sum(
        s.resources.gpus for s in result.stages
        if s.resources is not None and s.resources.gpus > 0
    )
    assert floor <= cluster.gpus + 1e-6, (
        f"Per-worker GPU floor {floor:.3f} exceeds cluster.gpus {cluster.gpus}; "
        f"streaming should not have been picked."
    )


def test_batch_mode_emits_only_gpu_fraction_for_full_class_stage(registry) -> None:
    """Batch mode still emits the right per-worker GPU fraction (VRAM
    packing); the backend decides how many actors to spawn."""

    cluster = ClusterProfile(cpus=64, gpus=4, gpu_memory_gb=24.0)
    result = tune(TunerInputs(
        stages=_refs("InferenceAsrNemoStage"),
        registry=registry,
        cluster=cluster,
        forced_execution_mode="batch",
    ))
    asr = result.stages[0]
    assert asr.resources is not None
    # 24 GB / 7.24 GB ≈ 3 → gpus_per_worker = 1/3
    assert asr.resources.gpus == pytest.approx(1.0 / 3, abs=1e-3)
    # And no num_workers pin — autoscaler handles it.
    assert asr.backend_hints is None or asr.backend_hints.num_workers is None


# ----------------------------------------------------------------------------
# Catalog coverage
# ----------------------------------------------------------------------------


def test_truth_table_covers_every_audio_stage_in_catalog(registry) -> None:
    """Every audio stage in the registry should be classified by the tuner."""

    missing: list[str] = []
    for name, entry in registry.by_name.items():
        target = entry.card.target
        if not (target.startswith("nemo_curator.stages.audio") or name == "ManifestReader"):
            continue
        if name not in STAGE_RESOURCE_PROFILES:
            missing.append(name)
    assert not missing, (
        "These audio stages are not classified by the tuner truth table; "
        "add them to STAGE_RESOURCE_PROFILES in nemo_curator/agentic/tuner.py: "
        f"{missing}"
    )


# ----------------------------------------------------------------------------
# Integration via plan_from_intent
# ----------------------------------------------------------------------------


def test_plan_from_intent_auto_decides_mode(registry) -> None:
    """plan_from_intent no longer accepts execution_mode; tuner picks it."""

    cluster = ClusterProfile(cpus=64, gpus=8, gpu_memory_gb=80.0)
    result = plan_from_intent(
        _tts_intent(),
        source_uri="/data/input.jsonl",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
        cluster=cluster,
    )
    assert result.ir.cluster == cluster
    assert result.ir.executor_config is not None
    assert result.ir.executor_config.execution_mode == "streaming"
    assert result.ir.executor_config.backend == "xenna"

    by_name = {s.stage: s for s in result.ir.stages}
    if "InferenceAsrNemoStage" in by_name:
        asr = by_name["InferenceAsrNemoStage"]
        assert asr.resources is not None and asr.resources.gpus > 0
        # backend_hints is intentionally None — autoscaler decides.
        assert asr.backend_hints is None or asr.backend_hints.num_workers is None


def test_plan_from_intent_falls_back_to_batch_on_tight_cluster(registry) -> None:
    """Small VRAM card → tuner picks batch automatically."""

    # 8 GB is tight enough that ASR's 7.24 GB need fills the card and
    # the streaming floor exceeds 1 GPU.
    cluster = ClusterProfile(cpus=16, gpus=1, gpu_memory_gb=8.0)
    result = plan_from_intent(
        _tts_intent(),
        source_uri="/data/input.jsonl",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
        cluster=cluster,
    )
    assert result.ir.executor_config.execution_mode == "batch"
    assert result.ir.executor == "ray_actor_pool"
