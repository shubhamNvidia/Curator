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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AudioForm = Literal["file", "waveform"]
ProducedForm = Literal["tensor", "disk"]
Cardinality = Literal["1:1", "1:1 nested-list", "1:N fan-out", "N:1", "filter"]
Dispatch = Literal["process", "process_batch", "auto"]
# How a stage handles per-item failures at runtime. "unknown" means the stage
# has not declared a uniform policy (the default for most stages today).
ErrorPolicy = Literal["skip", "fail", "annotate", "unknown"]


@dataclass(frozen=True)
class ParamSpec:
    """A single constructor parameter an agent can set on a stage.

    Usually derived automatically from the stage's dataclass fields via
    :func:`nemo_curator.stages.audio._agent_registry.stage_params`, but a stage
    may also override/augment entries in ``StageContract.params``.
    """

    name: str
    type: str = "Any"
    default: Any = None  # noqa: ANN401
    required: bool = False
    choices: list[Any] | None = None  # populated for Literal[...] params
    description: str | None = None


@dataclass(frozen=True)
class IOSpec:
    """Task data keys and audio forms read or written by a stage."""

    data_keys: list[str] = field(default_factory=list)
    segment_data_keys: list[str] = field(default_factory=list)
    accepts: list[AudioForm] = field(default_factory=list)
    produces: list[ProducedForm] = field(default_factory=list)


@dataclass(frozen=True)
class Gates:
    """Execution gates or side effects an agent should know before wrapping a stage."""

    writes_to_disk: bool = False
    requires_gpu: bool = False
    requires_internet_first_run: bool = False
    requires_ffmpeg: bool = False
    lifecycle_side_effects: bool = False
    runtime_secrets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SizeEnvelope:
    """Coarse size and memory hints for agent planning."""

    max_input_sec: float | None = None
    allowed_sample_rates: list[int] | None = None
    channels: Literal["mono", "stereo", "any"] = "any"
    memory_hint: str | None = None


@dataclass(frozen=True)
class StageContract:
    """Read-only discovery contract for an agent-ready processing stage."""

    reads: IOSpec = field(default_factory=IOSpec)
    writes: IOSpec = field(default_factory=IOSpec)
    reads_one_of: list[IOSpec] = field(default_factory=list)
    metadata_reads: list[str] = field(default_factory=list)
    metadata_writes: list[str] = field(default_factory=list)
    cardinality: Cardinality = "1:1"
    cardinality_options: list[str] = field(default_factory=list)
    iteration_key: str | None = None
    preserves_upstream_keys: bool = True
    wrappable: bool = True
    size_envelope: SizeEnvelope = field(default_factory=SizeEnvelope)
    gates: Gates = field(default_factory=Gates)
    # Agent-facing metadata (advisory; defaults keep older describe() calls valid).
    stage_id: str | None = None  # stable semantic id; defaults to the class name when None
    description: str | None = None  # one-line human summary for planners/UIs
    params: list[ParamSpec] = field(default_factory=list)  # usually auto-derived at discovery time
    dispatch: Dispatch = "auto"  # "auto" => infer from the stage at runtime
    error_policy: ErrorPolicy = "unknown"


class AgentReady:
    """Mixin for stages that expose a read-only agent discovery contract."""

    def describe(self) -> StageContract:
        raise NotImplementedError


def resolve_contract(stage: AgentReady | type) -> StageContract:
    """Return a stage's ``StageContract``.

    Accepts an instance (calls ``describe()``) or, for stages whose contract
    does not depend on instance state, a class (instantiated argument-free only
    when possible). Prefer passing an instance.
    """
    if isinstance(stage, type):
        msg = "resolve_contract requires a stage instance (describe() may depend on instance flags)"
        raise TypeError(msg)
    return stage.describe()
