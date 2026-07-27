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

"""Level-14 generalization proof: a brand-new stage + card the agent never saw
during development are (a) discoverable, (b) contract-derivable, and (c) accepted
by the conformance gate with zero drift — with NO change to the core.

    python eval/audio/fixtures/novel_stage/novel_card_check.py   # -> PASS
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_fixture_stage():
    """Import the fixture module so its stage registers via the metaclass."""
    path = os.path.join(_HERE, "new_loudness_norm.py")
    name = "new_loudness_norm_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # @dataclass resolves cls.__module__ via sys.modules
    spec.loader.exec_module(mod)  # metaclass registers NewLoudnessNormStage here
    return mod


def main() -> int:
    import yaml

    mod = _import_fixture_stage()
    from nemo_curator.audio_agent.card_conformance import check_card
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio._catalog import list_agent_ready_stages

    sid = "NewLoudnessNormStage"
    problems: list[str] = []

    # (a) discoverable through the normal catalog once imported
    discoverable = sid in list_agent_ready_stages()
    if not discoverable:
        problems.append(f"{sid} not discoverable via list_agent_ready_stages()")

    # (b) contract derivable by the same machinery used for shipped stages
    try:
        contract = foundation.build_contract(mod.NewLoudnessNormStage())
        writes = list(contract.writes.data_keys)
    except Exception as e:  # noqa: BLE001
        problems.append(f"build_contract failed: {type(e).__name__}: {e}")
        writes = []

    # (c) the conformance gate accepts the unseen card with zero drift
    card = yaml.safe_load(open(os.path.join(_HERE, "new_loudness_norm.yaml"), encoding="utf-8"))
    violations = check_card(sid, card)
    if violations:
        problems.append(f"check_card violations: {violations}")

    print(f"discoverable={discoverable} contract_writes={writes}")
    print(f"check_card violations={violations}")
    ok = not problems
    print("[novel_card_check]", "PASS" if ok else f"FAIL :: {problems}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
