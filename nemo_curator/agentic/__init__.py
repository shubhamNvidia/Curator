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
"""ADV Agentic Audio Curator — package root.

This subpackage implements the ADV agentic layer on top of NeMo Curator.
It is organized as follows:

- :mod:`nemo_curator.agentic.cards` — Pydantic schemas for stage / dataset /
  model / run cards. The shape of every machine-readable artifact the agent
  reasons over.
- :mod:`nemo_curator.agentic.intent` — :class:`IntentCategories` schema, the
  fixed JSON shape into which the LLM extracts a user prompt.
- :mod:`nemo_curator.agentic.registry` — :class:`CapabilityRegistry`. Scans
  the codebase + user plugins for stage cards and cross-checks against the
  in-process ``_STAGE_REGISTRY``.

Layers added in subsequent phases:

- :mod:`nemo_curator.agentic.ir` — Pipeline IR schema (Phase 2).
- :mod:`nemo_curator.agentic.validator` — Layer 2 static validator (Phase 2).
- :mod:`nemo_curator.agentic.compiler` — IR → canonical YAML (Phase 2).
- :mod:`nemo_curator.agentic.runner` — Execution + checkpoints (Phase 2).
- :mod:`nemo_curator.agentic.nat` — NAT integration (Phase 3).
- :mod:`nemo_curator.agentic.onboarding` — Wizard + wrapper API (Phase 4).
"""

from __future__ import annotations

__version__ = "0.1.0"
