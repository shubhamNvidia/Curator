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

"""Name -> class / target resolution over the Milestone-1 discovery catalog.

Every recipe ``ref`` is a registered agent-ready stage class name. We resolve it
through the existing foundation catalog so the recipe can only reference real,
importable stages (the anti-hallucination guarantee), and derive the Hydra
``_target_`` string for round-tripping to ``config.run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent_ready import StageContract


def resolve_stage_class(ref: str) -> type:
    """Return the registered stage class for ``ref`` (raises ``KeyError`` if unknown)."""
    from nemo_curator.stages.audio._catalog import get_agent_ready_stage_class

    return get_agent_ready_stage_class(ref)


def resolve_target(ref: str) -> str:
    """Return the fully qualified ``module.ClassName`` target for ``ref``."""
    cls = resolve_stage_class(ref)
    return f"{cls.__module__}.{cls.__qualname__}"


def static_contract_for(ref: str) -> StageContract:
    """Return the instance-free contract for ``ref``."""
    from nemo_curator.stages.audio._agent_registry import static_contract

    return static_contract(resolve_stage_class(ref))
