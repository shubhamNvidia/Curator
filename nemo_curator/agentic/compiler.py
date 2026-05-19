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
"""IR → canonical ``stages:`` YAML compiler.

The output matches the convention consumed by
``nemo_curator.config.run.create_pipeline_from_yaml``:

.. code-block:: yaml

    stages:
      - _target_: nemo_curator.stages.audio.common.GetAudioDurationStage
        audio_filepath_key: audio_filepath
        duration_key: duration
        resources:
          cpus: 1.0

Resources are emitted only when explicitly overridden by the IR; otherwise
each stage relies on its dataclass default. Stage cards' ``target`` field
provides the ``_target_`` for Hydra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nemo_curator.agentic.ir import PipelineIR, StageRef
from nemo_curator.agentic.registry import CapabilityRegistry


class CompileError(Exception):
    """Raised when an IR cannot be compiled."""


def compile_ir_to_yaml(ir: PipelineIR, registry: CapabilityRegistry) -> str:
    """Render the IR to a canonical YAML string."""

    payload: dict[str, Any] = {"stages": [_render_stage(s, registry) for s in ir.stages]}
    return yaml.safe_dump(payload, sort_keys=False)


def write_compiled_yaml(ir: PipelineIR, registry: CapabilityRegistry, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(compile_ir_to_yaml(ir, registry), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _render_stage(stage_ref: StageRef, registry: CapabilityRegistry) -> dict[str, Any]:
    entry = registry.get(stage_ref.stage)
    if entry is None:
        msg = f"compile: stage {stage_ref.stage!r} not in registry."
        raise CompileError(msg)
    target = entry.card.target
    out: dict[str, Any] = {"_target_": target}
    for k, v in stage_ref.params.items():
        if v is None:
            continue
        out[k] = v
    if stage_ref.resources is not None:
        out["resources"] = stage_ref.resources.to_resources_kwargs()
    return out


__all__ = ["CompileError", "compile_ir_to_yaml", "write_compiled_yaml"]
