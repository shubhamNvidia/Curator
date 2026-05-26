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
"""PipelineIR — the deterministic intermediate representation.

The IR is the *only* thing the planner ever produces. Every downstream layer
(validator, compiler, runner, critic, replay) reads the IR. The agent layer
itself eventually emits IR — until then, users write IR by hand.

Design notes:

- Each :class:`StageRef` names a stage by its **registry key** (the Python
  class name). The compiler maps ``stage`` → the StageCard's ``target`` for
  Hydra ``_target_``.
- The IR is intentionally linear (a list of stages). Composite stages are
  inlined by the compiler before flattening; we do not model DAGs at v1.
- :class:`SourceSpec` and :class:`SinkSpec` are explicit so the compiler can
  prepend / append the right reader / writer without having to second-guess
  user intent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_curator.agentic.cards import ResourceSpec
from nemo_curator.agentic.intent import IntentCategories


# ----------------------------------------------------------------------------
# Source / sink
# ----------------------------------------------------------------------------


class SourceSpec(BaseModel):
    """Where data comes from. The :mod:`adapters` module resolves this."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["manifest", "directory", "fleurs", "readspeech", "hf"] = "manifest"
    uri: str = Field(..., description="Local path, JSONL manifest, hf://..., etc.")
    file_extensions: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    options: dict[str, Any] = Field(default_factory=dict, description="Adapter-specific options.")


class SinkSpec(BaseModel):
    """Where outputs go."""

    model_config = ConfigDict(extra="forbid")

    target_dir: str = Field(..., description="Root of the run output.")
    audio_subdir: str = "audio"
    manifest_filename: str = "manifest.jsonl"
    output_format: Literal["wav", "flac", "ogg"] = "wav"
    overwrite: bool = False


# ----------------------------------------------------------------------------
# Stage references
# ----------------------------------------------------------------------------


class BackendHints(BaseModel):
    """Backend-neutral hints attached to a stage by the tuner.

    Each executor honours the subset it understands:

    - XennaExecutor reads ``num_workers``, ``num_workers_per_node``,
      ``slots_per_actor``, ``worker_max_lifetime_m``,
      ``worker_restart_interval_m``, ``ignore_failures``.
    - RayActorPoolExecutor reads ``num_workers`` (as an upper cap via
      :meth:`ProcessingStage.num_workers`).
    - RayDataExecutor reads ``num_workers``.

    Fields that a given backend doesn't understand are silently ignored —
    the IR stays portable.
    """

    model_config = ConfigDict(extra="forbid")

    num_workers: int | None = Field(default=None, ge=1)
    num_workers_per_node: int | None = Field(default=None, ge=1)
    slots_per_actor: int | None = Field(default=None, ge=1)
    worker_max_lifetime_m: int | None = Field(default=None, ge=1)
    worker_restart_interval_m: int | None = Field(default=None, ge=1)
    ignore_failures: bool | None = None


class StageRef(BaseModel):
    """One node in the IR — references a stage by class name and binds its params."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(
        default=None,
        description="Optional stable identifier the runner uses for checkpointing/cache keys.",
    )
    stage: str = Field(..., description="Stage class name (registry key).")
    params: dict[str, Any] = Field(default_factory=dict)
    resources: ResourceSpec | None = Field(
        default=None,
        description="Override the card's default resources (passed to .with_()).",
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Override the card's default batch_size (passed to .with_()).",
    )
    backend_hints: BackendHints | None = Field(
        default=None,
        description="Per-stage backend tuning hints emitted by the resource tuner.",
    )
    tuner_reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable why-chips for the tuner's decisions on this stage.",
    )
    notes: str | None = None
    auto_inserted: bool = Field(
        default=False,
        description="True for stages added by the validator's auto-insert layer.",
    )
    insert_reason: str | None = None


# ----------------------------------------------------------------------------
# Cluster + executor configuration
# ----------------------------------------------------------------------------


class ClusterProfile(BaseModel):
    """User-supplied cluster size the tuner allocates against.

    The web form collects these; the runner uses them to pick fractional
    GPU shares and worker counts that fit. The values are an *upper bound*
    — the executor's own resource discovery may still cap things lower at
    runtime (e.g. when other Ray jobs are sharing the cluster).
    """

    model_config = ConfigDict(extra="forbid")

    cpus: int = Field(default=8, ge=1, description="Total schedulable CPU cores.")
    gpus: int = Field(default=0, ge=0, description="Total schedulable GPUs.")
    gpu_memory_gb: float = Field(
        default=24.0,
        ge=0.0,
        description="Per-GPU memory in GB. Used to map gpu_memory_gb hints to fractional gpus.",
    )
    nodes: int = Field(default=1, ge=1)
    reserved_cpus: float = Field(default=0.0, ge=0.0)
    reserved_gpus: float = Field(default=0.0, ge=0.0)


class ExecutorConfig(BaseModel):
    """Top-level executor configuration emitted by the tuner.

    ``backend`` and ``execution_mode`` together pin the runtime behaviour:

    - ``backend="xenna"`` + ``execution_mode="streaming"`` — autoscaling,
      all stages run concurrently. GPU stages compete for the GPU pool.
    - ``backend="xenna"`` + ``execution_mode="batch"`` — Xenna in batch
      mode (drains each stage before the next).
    - ``backend="ray_actor_pool"`` — synchronous, one stage at a time,
      each stage owns the full cluster for its turn. ``execution_mode``
      is ignored.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["xenna", "ray_actor_pool", "ray_data"] = "xenna"
    execution_mode: Literal["streaming", "batch"] = "streaming"
    cpu_allocation_percentage: float = Field(default=0.95, ge=0.1, le=1.0)
    autoscale_interval_s: int = Field(default=180, ge=10)
    logging_interval_s: int = Field(default=60, ge=5)
    reserved_cpus: float = Field(default=0.0, ge=0.0)
    reserved_gpus: float = Field(default=0.0, ge=0.0)
    ignore_failures: bool = False
    tuner_reasons: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Top-level IR
# ----------------------------------------------------------------------------


class PipelineIR(BaseModel):
    """Top-level deterministic pipeline plan."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    name: str = "adv_pipeline"
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    source: SourceSpec
    sink: SinkSpec
    stages: list[StageRef] = Field(default_factory=list)

    intent: IntentCategories | None = None
    notes: list[str] = Field(default_factory=list)

    cluster: ClusterProfile | None = Field(
        default=None,
        description="User-supplied cluster size used by the resource tuner.",
    )
    executor_config: ExecutorConfig | None = Field(
        default=None,
        description="Executor selection + tuning emitted by the tuner.",
    )

    executor: Literal["xenna", "ray_data", "ray_actor_pool"] = "xenna"
    dry_run_sample_count: int = Field(default=4, ge=1, le=1024)
    enable_checkpointing: bool = True
    cache_enabled: bool = True

    @model_validator(mode="after")
    def _stage_ids_unique(self) -> PipelineIR:
        ids = [s.id for s in self.stages if s.id]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            msg = f"StageRef ids must be unique; duplicates: {dupes}"
            raise ValueError(msg)
        return self

    # ---- IO helpers ----------------------------------------------------

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_json(cls, text: str | bytes) -> PipelineIR:
        return cls.model_validate_json(text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineIR:
        return cls.model_validate(data)

    @classmethod
    def from_path(cls, path: str | Path) -> PipelineIR:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(text)
            return cls.model_validate(data)
        return cls.from_json(text)

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in {".yaml", ".yml"}:
            import yaml

            p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        else:
            p.write_text(self.to_json(), encoding="utf-8")


# ----------------------------------------------------------------------------
# Convenience builder for hand-authored tests
# ----------------------------------------------------------------------------


def stage(name: str, **params: Any) -> StageRef:
    """Compact constructor used in tests and few-shot example IRs."""

    return StageRef(stage=name, params=params)


__all__ = [
    "BackendHints",
    "ClusterProfile",
    "ExecutorConfig",
    "PipelineIR",
    "SinkSpec",
    "SourceSpec",
    "StageRef",
    "stage",
]
