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
"""MultiPipelineRunner — execute a fan-out lattice of IRs sequentially.

Today's use is mostly aspirational (augmentation variants land in Phase 5),
but shipping the runner now keeps the agent's compile target stable. Each
variant gets its own subdirectory under the shared ``target_dir`` and a
merge step concatenates the per-variant manifests with a ``variant`` column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nemo_curator.agentic.ir import PipelineIR
from nemo_curator.agentic.registry import CapabilityRegistry
from nemo_curator.agentic.runner import RunOptions, RunResult, run


@dataclass
class MultiPipelineSpec:
    """A named variant in the lattice."""

    name: str
    ir: PipelineIR


@dataclass
class MultiPipelineResult:
    results: dict[str, RunResult]
    merged_manifest_path: Path | None


def run_lattice(
    specs: list[MultiPipelineSpec],
    registry: CapabilityRegistry,
    *,
    options: RunOptions | None = None,
) -> MultiPipelineResult:
    """Run each spec in turn under a shared ``target_dir`` and merge manifests."""

    results: dict[str, RunResult] = {}
    base_dir: Path | None = None
    manifest_lines: list[dict] = []

    for spec in specs:
        sub_dir = Path(spec.ir.sink.target_dir).expanduser().resolve()
        sub_dir.mkdir(parents=True, exist_ok=True)
        if base_dir is None:
            base_dir = sub_dir.parent
        result = run(spec.ir, registry, options=options)
        results[spec.name] = result
        manifest = sub_dir / spec.ir.sink.manifest_filename
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row["variant"] = spec.name
                    manifest_lines.append(row)

    merged_path: Path | None = None
    if base_dir is not None and manifest_lines:
        merged_path = base_dir / "manifest.jsonl"
        with merged_path.open("w", encoding="utf-8") as f:
            for row in manifest_lines:
                f.write(json.dumps(row) + "\n")
        logger.info(f"multipipeline: merged {len(manifest_lines)} rows → {merged_path}")

    return MultiPipelineResult(results=results, merged_manifest_path=merged_path)


__all__ = ["MultiPipelineResult", "MultiPipelineSpec", "run_lattice"]
