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
"""Test-scope configuration for ``tests/agentic/``.

The repo-wide ``tests/conftest.py`` spins up a Ray cluster via an
``autouse=True`` session-scoped fixture. The agentic unit tests do not need
Ray and we want them to be runnable in any environment, so we override that
fixture with a no-op here.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> str:  # type: ignore[override]
    return "noop://agentic"
