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

"""Focused regressions for resource-planning and calibration safety."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_curator.audio_agent import _ray, calibration, planner, verbs
from nemo_curator.audio_agent.contracts import EnvProfile
from nemo_curator.audio_agent.report import _dedup_stage_perf
from nemo_curator.stages.audio.common import ManifestReader
from nemo_curator.stages.resources import Resources
from nemo_curator.utils.performance_utils import (
    StagePerfStats,
    resource_probe_metrics,
)


class _Index:
    def __init__(self, cards: dict[str, dict[str, Any]] | None = None):
        self.cards = cards or {}

    def card(self, name: str) -> dict[str, Any] | None:
        return self.cards.get(name)


class _Stage:
    def __init__(
        self,
        *,
        resources: Resources,
        num_workers: int | None = None,
        name: str = "runtime-stage",
    ):
        self.resources = resources
        self._num_workers = num_workers
        self.name = name

    def num_workers(self) -> int | None:
        return self._num_workers


def _contract(*, requires_gpu: bool = False) -> SimpleNamespace:
    return SimpleNamespace(gates=SimpleNamespace(requires_gpu=requires_gpu))


def _plan(
    stage: Any,  # noqa: ANN401
    env: EnvProfile,
    *,
    card: dict[str, Any] | None = None,
    calibration_facts: dict[str, Any] | None = None,
):
    return planner.plan(
        [stage],
        [_contract()],
        env,
        index=_Index({type(stage).__name__: card or {}}),
        calibration=calibration_facts,
    )


def test_cpu_demand_and_exact_ray_reservation_are_distinct() -> None:
    stage = _Stage(resources=Resources(cpus=3.5))
    result = _plan(
        stage,
        EnvProfile(total_cpus=4, total_ram_gb=64),
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
    )

    assert result.per_stage[0]["cpus"] == 1.0
    assert result.per_stage[0]["cpu_reservation"] == 3.5
    assert result.estimate["sum_cpu_demand"] == 1.0
    assert result.estimate["sum_cpu_reservation"] == 3.5
    assert result.estimate["allocatable_cpus"] == 3
    assert result.mode == "batch"
    assert result.feasible is False
    assert any("reserves 3.5 Ray CPU" in item for item in result.escalations)


def test_ray_cluster_capacity_replaces_driver_scheduling_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = EnvProfile(
        total_cpus=64,
        total_ram_gb=256,
        has_gpu=True,
        gpu_count=4,
        gpu_mem_gb=80,
        gpu_names=["driver-gpu"],
    )
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: {
            "CPU": 8.0,
            "GPU": 1.0,
            "memory": float(32 * 1024**3),
        },
    )

    result = verbs._apply_ray_cluster_capacity(
        env,
        "10.20.30.40:6379",
    )

    assert result.total_cpus == 8.0
    assert result.total_ram_gb == 32.0
    assert result.gpu_count == 1
    assert result.gpu_mem_gb == 0.0
    assert result.gpu_names == []
    assert any("bound to Ray cluster capacity" in note for note in result.notes)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        (None, True),
        ("", True),
        ("auto", True),
        ("local", True),
        ("localhost:6379", True),
        ("127.0.0.1:6379", True),
        ("127.99.1.2:6379", True),
        ("[::1]:6379", True),
        ("ray://localhost:10001", True),
        ("ray://[::1]:10001", True),
        ("10.20.30.40:6379", False),
        ("ray://remote.example:10001", False),
        ("0.0.0.0:6379", False),
    ],
)
def test_ray_address_locality_is_deterministic(
    address: str | None,
    expected: bool,
) -> None:
    assert verbs._ray_address_is_local(address) is expected


def test_loopback_ray_capacity_keeps_local_driver_gpu_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = EnvProfile(
        total_cpus=64,
        total_ram_gb=256,
        has_gpu=True,
        gpu_count=1,
        gpu_mem_gb=80,
        gpu_names=["local-driver-gpu"],
    )
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: {
            "CPU": 8.0,
            "GPU": 1.0,
            "memory": float(32 * 1024**3),
        },
    )

    result = verbs._apply_ray_cluster_capacity(
        env,
        "ray://[::1]:10001",
    )

    assert result.gpu_mem_gb == 80
    assert result.gpu_names == ["local-driver-gpu"]


def test_ray_cluster_probe_failure_does_not_fall_back_to_driver_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = EnvProfile(total_cpus=64, total_ram_gb=256, gpu_count=4)
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: (_ for _ in ()).throw(
            RuntimeError("remote cluster unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="refusing to substitute driver resources"):
        verbs._apply_ray_cluster_capacity(env, "10.20.30.40:6379")


def test_ray_cluster_with_no_cpu_capacity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: {"CPU": 0.0, "GPU": 1.0},
    )

    with pytest.raises(RuntimeError, match="no positive finite CPU capacity"):
        verbs._apply_ray_cluster_capacity(
            EnvProfile(total_cpus=64, total_ram_gb=256),
            "10.20.30.40:6379",
        )


def test_custom_executor_never_inherits_driver_capacity() -> None:
    driver = EnvProfile(
        has_gpu=True,
        gpu_count=4,
        gpu_mem_gb=80,
        gpu_names=["driver-only"],
        total_cpus=64,
        total_ram_gb=256,
    )

    target = verbs._resource_environment(driver, None, "custom_executor")

    assert target.has_gpu is False
    assert target.gpu_count == 0
    assert target.gpu_mem_gb == 0
    assert target.total_cpus == 0
    assert driver.gpu_count == 4


def test_custom_executor_owns_capacity_and_bounded_remote_smoke_can_probe_vram() -> None:
    custom = SimpleNamespace(
        feasible=False,
        escalations=["a stage reserves 1 GPU but no GPU is available"],
        notes=[],
    )
    remote_smoke = SimpleNamespace(
        feasible=False,
        escalations=["GPU VRAM capacity is unknown; cannot prove a stage needing 8 GB fits"],
        notes=[],
    )
    remote_run = SimpleNamespace(
        feasible=False,
        escalations=["GPU VRAM capacity is unknown; cannot prove a stage needing 8 GB fits"],
        notes=[],
    )

    verbs._adapt_resource_plan_for_target(
        custom,
        execution_target="custom_executor",
        operation="run",
    )
    verbs._adapt_resource_plan_for_target(
        remote_smoke,
        execution_target="external_ray",
        operation="smoke",
    )
    verbs._adapt_resource_plan_for_target(
        remote_run,
        execution_target="external_ray",
        operation="run",
    )

    assert custom.feasible is True
    assert custom.escalations == []
    assert remote_smoke.feasible is True
    assert remote_smoke.escalations == []
    assert remote_run.feasible is False
    assert remote_run.escalations


def test_smoke_stops_only_the_ray_head_it_bootstrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    stopped: list[str] = []
    ownership_checks = 0

    def owns_after_bootstrap(address=None):
        nonlocal ownership_checks
        ownership_checks += 1
        return ownership_checks > 1 and address in {
            None,
            "127.0.0.1:62000",
        }

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "_bootstrap_ray", lambda: "127.0.0.1:62000")
    monkeypatch.setattr(_ray, "owns_cluster", owns_after_bootstrap)
    monkeypatch.setattr(
        _ray,
        "shutdown_cluster",
        lambda address=None: stopped.append(str(address)) or True,
    )
    monkeypatch.setattr(verbs, "_apply_ray_cluster_capacity", lambda env, _address: env)
    monkeypatch.setattr(verbs, "probe_env", lambda: EnvProfile(total_cpus=8, total_ram_gb=32))
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            feasible=True,
            escalations=[],
            machine_fingerprint="machine",
        ),
    )
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: ([], "batch"),
    )

    result = verbs.smoke(
        {
            "stages": [
                {
                    "ref": "ManifestReader",
                    "params": {"manifest_path": str(source)},
                }
            ]
        },
        sample=1,
        bootstrap_ray=True,
    )

    assert result["status"] == "completed"
    assert stopped == ["127.0.0.1:62000"]
    assert "ray_bootstrap_cleanup=completed" in result["notes"]


def test_fixed_workers_multiply_cpu_and_gpu_scheduling_footprints() -> None:
    stage = _Stage(resources=Resources(cpus=0.5, gpus=0.5), num_workers=3)
    result = _plan(
        stage,
        EnvProfile(
            has_gpu=True,
            gpu_count=1,
            gpu_mem_gb=24,
            total_cpus=8,
            total_ram_gb=64,
        ),
        card={
            "resource": {
                "cpus": 0.5,
                "gpu_mem_gb": 1.0,
                "host_mem_gb": 1.0,
                "gpu_optional": True,
            }
        },
    )

    assert result.per_stage[0]["num_workers"] == 3
    assert result.estimate["sum_cpu_reservation"] == 1.5
    assert result.estimate["sum_gpu_reservation"] == 1.5
    assert result.mode == "batch"
    assert result.feasible is False
    assert any("reserves 1.5 Ray GPU" in item for item in result.escalations)


def test_positive_gpu_reservation_needs_a_gpu_even_when_card_says_optional() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=0.1))
    result = _plan(
        stage,
        EnvProfile(gpu_count=0, total_cpus=8, total_ram_gb=64),
        card={
            "resource": {
                "cpus": 1.0,
                "gpu_mem_gb": 0.1,
                "host_mem_gb": 1.0,
                "gpu_optional": True,
            }
        },
    )

    assert result.feasible is False
    assert any("but no GPU is available" in item for item in result.escalations)


def test_composite_uses_configured_child_resources_and_workers() -> None:
    stage = ManifestReader(manifest_path="/tmp/input.jsonl")
    stage.with_(
        {
            "file_partitioning": {
                "resources": Resources(cpus=2.0),
                "num_workers": 2,
            }
        }
    )
    operations_before = deepcopy(stage._with_operations)

    result = planner.plan(
        [stage],
        [_contract()],
        EnvProfile(total_cpus=4, total_ram_gb=64),
        index=_Index(),
    )

    assert result.mode == "batch"
    assert result.feasible is False
    assert result.estimate["sum_cpu_reservation"] == 5.0
    assert any("expanded" in note for note in result.notes)
    assert stage._with_operations == operations_before


def test_planning_does_not_mutate_stage_resources_or_calibration() -> None:
    resources = Resources(cpus=1.25, gpus=0.25)
    stage = _Stage(resources=resources, num_workers=2)
    env = EnvProfile(
        has_gpu=True,
        gpu_count=2,
        gpu_mem_gb=24,
        total_cpus=16,
        total_ram_gb=64,
    )
    facts = {
        "calibration": {
            "_Stage": {
                "cpus": 1.5,
                "host_mem_gb": 2.0,
                "source": "measured",
                "machine_fingerprint": env.fingerprint(),
            }
        }
    }
    resources_before = deepcopy(resources)
    facts_before = deepcopy(facts)

    _plan(stage, env, calibration_facts=facts)

    assert stage.resources is resources
    assert stage.resources == resources_before
    assert facts == facts_before


def test_calibrate_wrapper_is_accepted_on_matching_machine() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    env = EnvProfile(total_cpus=8, total_ram_gb=32)
    result = _plan(
        stage,
        env,
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
        calibration_facts={
            "machine_fingerprint": env.fingerprint(),
            "calibration": {
                "_Stage": {
                    "cpus": 2.5,
                    "host_mem_gb": 3.0,
                    "source": "measured",
                }
            },
        },
    )

    assert result.per_stage[0]["cpus"] == 2.5
    assert result.per_stage[0]["host_mem_gb"] == 3.0
    assert result.per_stage[0]["source"] == "measured"


def test_bounded_calibration_cannot_lower_card_resource_estimates() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    env = EnvProfile(total_cpus=16, total_ram_gb=64, gpu_count=1, gpu_mem_gb=24)
    result = _plan(
        stage,
        env,
        card={
            "resource": {
                "cpus": 4.0,
                "host_mem_gb": 8.0,
                "gpu_mem_gb": 12.0,
            }
        },
        calibration_facts={
            "_Stage": {
                "cpus": 1.0,
                "host_mem_gb": 2.0,
                "gpu_mem_gb": 3.0,
                "source": "measured",
                "machine_fingerprint": env.fingerprint(),
            }
        },
    )

    assert result.per_stage[0]["cpus"] == 4.0
    assert result.per_stage[0]["host_mem_gb"] == 8.0
    assert result.per_stage[0]["gpu_mem_gb"] == 12.0
    assert result.per_stage[0]["source"] == "card"
    assert result.per_stage[0]["resource_sources"] == {
        "cpus": "card",
        "gpu_mem_gb": "card",
        "host_mem_gb": "card",
    }
    assert not any("using measured calibration" in note for note in result.notes)


def test_calibration_provenance_only_marks_resources_it_actually_raised() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=1))
    env = EnvProfile(
        total_cpus=16,
        total_ram_gb=64,
        gpu_count=1,
        gpu_mem_gb=24,
    )
    result = _plan(
        stage,
        env,
        card={
            "resource": {
                "cpus": 4.0,
                "host_mem_gb": 8.0,
                "gpu_mem_gb": 12.0,
                "gpu_optional": False,
            }
        },
        calibration_facts={
            "_Stage": {
                "cpus": 6.0,
                "host_mem_gb": 2.0,
                "gpu_mem_gb": 12.0,
                "source": "measured",
                "machine_fingerprint": env.fingerprint(),
            }
        },
    )

    assert result.per_stage[0]["source"] == "measured"
    assert result.per_stage[0]["resource_sources"] == {
        "cpus": "measured",
        "gpu_mem_gb": "card",
        "host_mem_gb": "card",
    }
    assert any("using measured calibration for 1 stage" in note for note in result.notes)


def test_unknown_vram_is_advisory_until_smoke_measures_the_real_fit() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=1))
    result = _plan(
        stage,
        EnvProfile(
            total_cpus=8,
            total_ram_gb=64,
            gpu_count=1,
            gpu_mem_gb=0,
        ),
        card={
            "resource": {
                "cpus": 1.0,
                "host_mem_gb": 1.0,
                "gpu_mem_gb": 8.0,
                "gpu_optional": False,
            }
        },
    )

    assert result.mode == "streaming"
    assert result.feasible is True
    assert result.estimate["gpu_mem_known"] is False
    assert any("GPU VRAM for this machine is unknown" in item for item in result.notes)
    assert not any("VRAM" in item for item in result.escalations)


def test_unknown_vram_does_not_block_gpu_optional_cpu_execution() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=0))
    result = _plan(
        stage,
        EnvProfile(
            total_cpus=8,
            total_ram_gb=64,
            gpu_count=1,
            gpu_mem_gb=0,
        ),
        card={
            "resource": {
                "cpus": 1.0,
                "host_mem_gb": 1.0,
                "gpu_mem_gb": 8.0,
                "gpu_optional": True,
            }
        },
    )

    assert result.mode == "streaming"
    assert result.feasible is True
    assert result.estimate["sum_gpu_mem_gb"] == 0


def test_mismatched_machine_calibration_is_ignored() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    result = _plan(
        stage,
        EnvProfile(total_cpus=8, total_ram_gb=32),
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
        calibration_facts={
            "_Stage": {
                "cpus": 7.0,
                "source": "measured",
                "machine_fingerprint": "another-machine",
            }
        },
    )

    assert result.per_stage[0]["cpus"] == 1.0
    assert result.per_stage[0]["source"] == "card"
    assert any("does not match this machine" in note for note in result.notes)


def test_nonfinite_and_negative_calibration_values_are_ignored() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    result = _plan(
        stage,
        EnvProfile(total_cpus=8, total_ram_gb=32),
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
        calibration_facts={
            "_Stage": {
                "cpus": float("inf"),
                "host_mem_gb": -1,
                "source": "measured",
            }
        },
    )

    assert result.per_stage[0]["cpus"] == 1.0
    assert result.per_stage[0]["host_mem_gb"] == 1.0
    assert result.per_stage[0]["source"] == "card"
    assert sum("finite non-negative" in note for note in result.notes) == 2


def test_perf_aggregates_include_extrema_and_calibration_uses_peak() -> None:
    first = StagePerfStats(
        stage_name="scorer",
        custom_metrics={"peak_vram_gb": 1.0, "throughput": 2.0},
    )
    second = StagePerfStats(
        stage_name="scorer",
        custom_metrics={"peak_vram_gb": 3.0, "throughput": 4.0},
    )
    metrics = _dedup_stage_perf(
        [
            SimpleNamespace(_stage_perf=[first]),
            SimpleNamespace(_stage_perf=[second]),
        ]
    )

    assert metrics["scorer"]["custom.peak_vram_gb"] == {
        "sum": 4.0,
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
        "count": 2,
    }
    measured = calibration.from_smoke({"per_stage_metrics": metrics})
    assert measured["scorer"]["gpu_mem_gb"] == 3.0
    assert measured["scorer"]["throughput"] == 3.0


def test_runtime_resource_probe_produces_host_memory_and_throughput() -> None:
    metrics = resource_probe_metrics(
        gpu_probe_started=False,
        process_time=2.0,
        num_items=5,
    )

    assert metrics["throughput"] == 2.5
    assert metrics["peak_host_mem_gb"] > 0
    assert "peak_vram_gb" not in metrics


def test_calibration_extraction_ignores_invalid_measurements() -> None:
    measured = calibration.from_smoke(
        {
            "per_stage_metrics": {
                "scorer": {
                    "peak_vram_gb": {"max": float("nan")},
                    "peak_host_mem_gb": {"max": -1.0},
                    "throughput": {"mean": float("inf")},
                }
            }
        }
    )

    assert measured == {}


def test_recalibrating_a_saved_smoke_preserves_its_machine_fingerprint() -> None:
    measured = calibration.from_smoke(
        {
            "per_stage_metrics": {
                "scorer": {
                    "peak_vram_gb": {"max": 3.0},
                }
            },
            "calibration": {
                "prior-stage": {
                    "gpu_mem_gb": 1.0,
                    "source": "measured",
                    "machine_fingerprint": "machine-A",
                }
            },
        }
    )

    assert measured["scorer"]["machine_fingerprint"] == "machine-A"


def test_env_fingerprint_changes_with_total_ram() -> None:
    base = EnvProfile(total_cpus=8, total_ram_gb=32)
    more_ram = EnvProfile(total_cpus=8, total_ram_gb=64)

    assert base.fingerprint() != more_ram.fingerprint()
