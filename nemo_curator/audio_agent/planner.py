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

"""Deterministic resource planner (1C.1): pick execution mode + feasibility.

Given each stage's declared needs (capability-card ``resource`` facts when present,
else conservative defaults) and the machine (:class:`EnvProfile`), decide
**streaming vs batch** by a feasibility check over CPU / GPU-memory / host-RAM
(+ disk headroom), defaulting to streaming and falling back to batch. The result
is a :class:`ResourcePlan` attached to the recipe as a recomputable annotation
(layered save, 1.2).

Scope of 1C.1: mode selection + feasibility + escalation. Per-stage resource
*assignment* (1C.2) and measured calibration (the ``data_driven`` module) refine
these numbers later; this module never reimplements Xenna's bin-packer.

GPU feasibility tracks TWO INDEPENDENT constraints (both must hold):
  (a) Ray reservation (scheduling) -- each stage reserves ``resources.gpus`` (the
      fraction Xenna pins per replica); the concurrent sum must fit the GPU COUNT,
      else Xenna aborts ("requires 1.5 but only 1 are available").
  (b) VRAM fit (memory) -- the concurrent VRAM (card/measured ``gpu_mem_gb``) must
      fit one GPU's memory when the machine VRAM is known.

    streaming feasible iff  Sum(cpus) <= cpus*0.95  AND  Sum(host_mem) <= ram*0.90
                            AND  Sum(gpu_reservation) <= num_gpus
                            AND  Sum(gpu_mem_gb) <= machine_gpu_mem
    else batch (only the largest single stage must fit each dimension:
    max(gpu_reservation) <= num_gpus AND max(gpu_mem_gb) <= machine_gpu_mem);
    if even that fails -> escalate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.audio_agent.contracts import EnvProfile

CPU_ALLOC = 0.95  # cpu_allocation_percentage headroom
RAM_ALLOC = 0.90  # host-RAM headroom
_DEFAULT_GPU_MEM_GB = 2.0  # conservative VRAM estimate for a GPU stage without a card fact
_DEFAULT_HOST_MEM_GB = 1.0  # conservative host-RAM estimate per stage without a card fact


@dataclass
class StageNeed:
    """Per-stage resource needs (absolute, machine-independent)."""

    index: int
    name: str
    cpus: float
    gpu_mem_gb: float
    host_mem_gb: float
    gpu_optional: bool = True
    # Ray ``Resources.gpus`` the stage reserves for SCHEDULING (the fraction Xenna
    # pins per replica) -- independent of ``gpu_mem_gb`` (the VRAM it actually needs).
    gpu_reservation: float = 0.0
    source: str = "default"  # measured (calibration) | card | default


@dataclass
class ResourcePlan:
    """The planner's output: mode + per-stage needs + feasibility + estimate."""

    mode: str = "streaming"  # "streaming" | "batch"
    feasible: bool = True
    per_stage: list[dict[str, Any]] = field(default_factory=list)
    estimate: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    machine_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feasible": self.feasible,
            "per_stage": self.per_stage,
            "estimate": self.estimate,
            "notes": self.notes,
            "escalations": self.escalations,
            "machine_fingerprint": self.machine_fingerprint,
        }


def _gpu_fraction(gpu_mem_gb: float, machine_gpu_mem_gb: float) -> float:
    """VRAM need as a fraction of one GPU; 1.0 (a whole GPU) when the size is unknown."""
    if gpu_mem_gb <= 0:
        return 0.0
    if machine_gpu_mem_gb > 0:
        return gpu_mem_gb / machine_gpu_mem_gb
    return 1.0  # GPU needed but machine VRAM unknown -> assume a whole GPU (conservative)


def _stage_need(
    index: int, stage: Any, contract: Any, card: dict[str, Any] | None, calib: dict[str, Any] | None = None  # noqa: ANN401
) -> StageNeed:
    """Derive a stage's absolute needs, preferring measured calibration > card > default.

    ``calib`` (from a prior smoke, 1C.2) overrides card ``resource`` facts field by
    field, so a stage whose real VRAM was measured plans against the measured number
    instead of the card's ``best_guess``.
    """
    name = type(stage).__name__
    res = (card or {}).get("resource", {}) or {}
    calib = calib or {}
    requires_gpu = bool(getattr(getattr(contract, "gates", None), "requires_gpu", False))

    def pick(key: str, card_default: Any) -> Any:  # measured > card > default  # noqa: ANN401
        if key in calib and calib[key] is not None:
            return calib[key]
        return res.get(key, card_default)

    cpus = float(pick("cpus", 1.0))
    gpu_mem = pick("gpu_mem_gb", _DEFAULT_GPU_MEM_GB if requires_gpu else 0.0)
    host_mem = float(pick("host_mem_gb", _DEFAULT_HOST_MEM_GB))
    gpu_optional = bool(res.get("gpu_optional", True))
    # The Ray GPU reservation is a scheduling fact only the built stage knows
    # (``resources.gpus``); it is NOT a card/calibration number.
    gpu_reservation = float(getattr(getattr(stage, "resources", None), "gpus", 0.0) or 0.0)
    source = "measured" if calib.get("source") == "measured" and any(k in calib for k in ("cpus", "gpu_mem_gb", "host_mem_gb")) else ("card" if res else "default")
    return StageNeed(index, name, cpus, float(gpu_mem or 0.0), host_mem, gpu_optional, gpu_reservation, source=source)


def plan(
    stages: list[Any],
    contracts: list[Any],
    env: EnvProfile,
    data_profile: dict[str, Any] | None = None,
    *,
    index: Any = None,  # noqa: ANN401
    calibration: dict[str, Any] | None = None,
) -> ResourcePlan:
    """Choose execution mode + report feasibility for ``stages`` on ``env``.

    Default streaming; fall back to batch when the concurrent sum doesn't fit
    (CPU, host-RAM, Ray GPU reservations, or VRAM). If even the largest single
    stage doesn't fit (reservation > num_gpus, VRAM > machine VRAM, or a GPU-only
    stage has no GPU), mark ``feasible=False`` with escalations. Per-stage needs
    prefer measured ``calibration`` (1C.2, from a prior smoke) over card
    ``resource`` facts over conservative defaults; the Ray GPU *reservation* is
    always read from the built stage's ``resources.gpus``.
    """
    from nemo_curator.audio_agent.index import get_index

    idx = index or get_index()
    calibration = calibration or {}

    def _calib_for(st: Any) -> dict[str, Any] | None:  # noqa: ANN401
        # Match by class name (manual calibration dicts) OR stage.name (what a smoke's
        # per_stage_metrics / from_smoke key by, e.g. "UTMOSFilter" vs "UTMOSFilterStage").
        return calibration.get(type(st).__name__) or calibration.get(getattr(st, "name", "") or "\0")

    needs = [
        _stage_need(i, st, contracts[i], idx.card(type(st).__name__), _calib_for(st))
        for i, st in enumerate(stages)
    ]
    if any(n.source == "measured" for n in needs):
        rp_note = f"using measured calibration for {sum(1 for n in needs if n.source == 'measured')} stage(s)"
    else:
        rp_note = ""

    total_cpus = float(env.total_cpus or 1)
    num_gpus = int(env.gpu_count or 0)
    machine_gpu_mem = float(env.gpu_mem_gb or 0.0)
    total_ram = float(env.total_ram_gb or 0.0)

    sum_cpus = sum(n.cpus for n in needs)
    sum_ram = sum(n.host_mem_gb for n in needs)
    max_cpus = max((n.cpus for n in needs), default=0.0)
    max_ram = max((n.host_mem_gb for n in needs), default=0.0)
    # GPU is TWO INDEPENDENT constraints, tracked separately (both must hold):
    #  (a) Ray reservation (scheduling): stages reserve ``resources.gpus``; Xenna pins
    #      that fraction per replica, so the CONCURRENT sum must fit the GPU COUNT (else
    #      it aborts, e.g. "requires 1.5 but only 1 are available").
    #  (b) VRAM fit (memory): the CONCURRENT VRAM must fit one GPU's memory (when known).
    sum_reservation = sum(n.gpu_reservation for n in needs)
    max_reservation = max((n.gpu_reservation for n in needs), default=0.0)
    sum_gpu_mem = sum(n.gpu_mem_gb for n in needs)
    max_gpu_mem = max((n.gpu_mem_gb for n in needs), default=0.0)
    sum_gpu_fraction = sum(_gpu_fraction(n.gpu_mem_gb, machine_gpu_mem) for n in needs)  # informational

    rp = ResourcePlan(machine_fingerprint=env.fingerprint())
    if rp_note:
        rp.notes.append(rp_note)

    cpu_ok = sum_cpus <= total_cpus * CPU_ALLOC
    ram_ok = total_ram <= 0 or sum_ram <= total_ram * RAM_ALLOC  # unknown RAM -> don't block
    reservation_stream_ok = sum_reservation <= num_gpus  # (a) concurrent Ray reservations fit the GPU count
    vram_stream_ok = machine_gpu_mem <= 0 or sum_gpu_mem <= machine_gpu_mem  # (b) concurrent VRAM fits

    if cpu_ok and ram_ok and reservation_stream_ok and vram_stream_ok:
        rp.mode = "streaming"
    else:
        rp.mode = "batch"
        rp.notes.append(
            f"streaming does not fit (cpu_ok={cpu_ok}, ram_ok={ram_ok}, "
            f"gpu_reservation_ok={reservation_stream_ok} [sum {round(sum_reservation, 2)} vs {num_gpus} GPU(s)], "
            f"gpu_vram_ok={vram_stream_ok} [sum {round(sum_gpu_mem, 2)} vs {machine_gpu_mem} GB]); "
            f"falling back to batch (sequential)"
        )

    # Batch feasibility: only the largest single stage must fit each dimension; else escalate.
    if rp.mode == "batch":
        if max_cpus > total_cpus:
            rp.feasible = False
            rp.escalations.append(f"a single stage needs {max_cpus} CPUs > machine {total_cpus}")
        if num_gpus == 0 and any((n.gpu_reservation > 0 or n.gpu_mem_gb > 0) and not n.gpu_optional for n in needs):
            rp.feasible = False
            rp.escalations.append("a GPU-only stage requires a GPU but none is available")
        if num_gpus > 0 and max_reservation > num_gpus:
            rp.feasible = False
            rp.escalations.append(f"a single stage reserves {max_reservation} Ray GPU(s) > available {num_gpus}")
        if machine_gpu_mem > 0 and max_gpu_mem > machine_gpu_mem:
            rp.feasible = False
            rp.escalations.append(f"a single stage needs {max_gpu_mem} GB VRAM > machine GPU memory {machine_gpu_mem} GB")
        if total_ram > 0 and max_ram > total_ram:
            rp.feasible = False
            rp.escalations.append(f"a single stage needs {max_ram} GB RAM > machine {total_ram} GB")

    # Disk headroom (best-effort; per-file sizing is a data_driven refinement).
    if env.free_disk_gb and env.free_disk_gb < 1.0:
        rp.notes.append(f"low free disk ({env.free_disk_gb} GB) - outputs may not fit")

    rp.per_stage = [
        {"stage_index": n.index, "stage": n.name, "cpus": n.cpus,
         "gpu_reservation": n.gpu_reservation, "gpu_mem_gb": n.gpu_mem_gb,
         "host_mem_gb": n.host_mem_gb, "source": n.source}
        for n in needs
    ]
    rp.estimate = {
        "mode": rp.mode,
        "num_files": int((data_profile or {}).get("num_files", 0)),
        "sum_cpus": round(sum_cpus, 2),
        "sum_gpu_reservation": round(sum_reservation, 2),
        "max_gpu_reservation": round(max_reservation, 2),
        "sum_gpu_mem_gb": round(sum_gpu_mem, 2),
        "sum_gpu_fraction": round(sum_gpu_fraction, 2),
        "sum_host_mem_gb": round(sum_ram, 2),
        "machine": {"cpus": total_cpus, "gpus": num_gpus, "gpu_mem_gb": machine_gpu_mem, "ram_gb": total_ram},
    }
    return rp
