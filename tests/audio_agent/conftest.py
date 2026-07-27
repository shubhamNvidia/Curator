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

"""Local test config for the audio_agent unit tests.

The repo-root ``tests/conftest.py`` declares a *session-scoped, autouse*
``shared_ray_cluster`` fixture, so every test transitively starts (and waits on)
a real Ray cluster. The audio_agent unit tests are **pure logic** — redaction and
token math (``test_safety``), recipe validation (``test_validate``), verb gates
(``test_verbs``), resource planning (``test_planner``), acceptance math
(``test_acceptance``), and continuation planning (``test_continuation``). None of
them execute a pipeline, so requiring Ray only makes them slow and unrunnable in
environments where Ray can't come up.

Overriding the fixture here (nearest-conftest wins) makes these tests start no Ray
cluster when run on their own (``pytest tests/audio_agent``), so they run fast and
anywhere. A test that genuinely needs Ray should request ``shared_ray_client``
explicitly and live outside this directory (or start its own cluster).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> Iterator[str]:
    """No-op override of the repo-root Ray fixture: these are Ray-free unit tests."""
    yield "audio_agent-unit-tests://no-ray"
