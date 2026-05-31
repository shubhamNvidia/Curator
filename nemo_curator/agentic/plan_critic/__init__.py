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
"""Build-time critic framework.

Two specialized critics share one ``CriticFinding`` contract:

- :class:`SanityCritic` (pure Python) catches structural bugs the
  deterministic planner couldn't have known about (missing outputs,
  contradictory intent, useless stages, resource over/under-allocation).
- :class:`PlanCritic` (LLM-driven) reviews the *intent* of the compiled
  pipeline against the user's prompt and the dataset profile, then
  proposes intent patches that the planner can re-apply.

A separate **runtime critic** lives in :mod:`nemo_curator.agentic.critic`
and reviews execution metrics after a pipeline run. That critic has a
different input shape (RunCard, score distributions, sample listening)
and therefore its own module.
"""

from nemo_curator.agentic.plan_critic.base import (
    BaseCritic,
    CriticFinding,
    CriticReport,
    CriticSeverity,
    apply_findings,
)
from nemo_curator.agentic.plan_critic.orchestrator import (
    ReviewResult,
    default_critics,
    review_and_replan,
)
from nemo_curator.agentic.plan_critic.plan import PlanCritic
from nemo_curator.agentic.plan_critic.sanity import SanityCritic

__all__ = [
    "BaseCritic",
    "CriticFinding",
    "CriticReport",
    "CriticSeverity",
    "PlanCritic",
    "ReviewResult",
    "SanityCritic",
    "apply_findings",
    "default_critics",
    "review_and_replan",
]
