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
"""NVIDIA NeMo Agent Toolkit integration for ADV Agentic.

Importing this package triggers the registration of every agentic tool with
NAT's global function registry. The actual implementations live in
:mod:`nemo_curator.agentic.tools` (NAT-independent); this package wires them
into NAT.

Typical use::

    nat run --config_file curator-adv.yaml \\
            --input "Build me a clean single-speaker dataset from /data/audio."

See ``workflow.yml`` in this directory for a complete reference config.
"""

from __future__ import annotations

# Eagerly load the registration module so tools are visible to the NAT
# discovery pass triggered by ``nat run``.
from nemo_curator.agentic.nat import register as _register  # noqa: F401
