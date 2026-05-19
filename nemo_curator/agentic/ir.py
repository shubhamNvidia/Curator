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
    notes: str | None = None
    auto_inserted: bool = Field(
        default=False,
        description="True for stages added by the validator's auto-insert layer.",
    )
    insert_reason: str | None = None


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


__all__ = ["PipelineIR", "SinkSpec", "SourceSpec", "StageRef", "stage"]
