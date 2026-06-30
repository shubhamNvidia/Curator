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

"""Discovery API for agent-ready audio stages.

Built on the framework's existing class-name registry
(:data:`nemo_curator.stages.base._STAGE_REGISTRY`, populated by ``StageMeta``);
no new registry is introduced. ``_ensure_audio_stages_imported`` triggers the
audio stage modules so their registration is populated before discovery.

    list_agent_ready_stages()         # -> ["MonoConversionStage", ...]
    describe_stage("UTMOSFilterStage")  # -> StageContract (static, instance-free)
    catalog_as_json()                 # -> JSON an agent/UI can consume
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import warnings
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio._agent_ready import AgentReady
from nemo_curator.stages.audio._agent_registry import build_contract, static_contract

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent_ready import StageContract

_IMPORTED = False


def _ensure_audio_stages_imported() -> None:
    """Import audio stage modules so ``StageMeta`` has registered their classes.

    Defensive: a submodule whose optional heavy dependency (whisperx, pyannote,
    nemo_text_processing, ...) is absent is skipped with a warning rather than
    breaking discovery. Idempotent.
    """
    global _IMPORTED  # noqa: PLW0603
    if _IMPORTED:
        return
    import nemo_curator.stages.audio as audio_pkg

    for modinfo in pkgutil.walk_packages(audio_pkg.__path__, prefix=audio_pkg.__name__ + "."):
        leaf = modinfo.name.rsplit(".", 1)[-1]
        if leaf.startswith("_"):  # private support modules carry no stages
            continue
        try:
            importlib.import_module(modinfo.name)
        except Exception as e:  # noqa: BLE001 - optional dep or import-time issue; skip
            warnings.warn(f"audio catalog: skipped {modinfo.name} ({type(e).__name__}: {e})", stacklevel=2)
    _IMPORTED = True


def _agent_ready_registry() -> dict[str, type]:
    from nemo_curator.stages.base import _STAGE_REGISTRY

    return {
        name: cls
        for name, cls in _STAGE_REGISTRY.items()
        if isinstance(cls, type) and issubclass(cls, AgentReady)
    }


def list_agent_ready_stages() -> list[str]:
    """Sorted class names of all registered agent-ready audio stages."""
    _ensure_audio_stages_imported()
    return sorted(_agent_ready_registry())


def get_agent_ready_stage_class(name: str) -> type:
    """Return the registered stage class for ``name`` (must be agent-ready)."""
    _ensure_audio_stages_imported()
    registry = _agent_ready_registry()
    if name not in registry:
        msg = f"{name!r} is not a registered agent-ready audio stage"
        raise KeyError(msg)
    return registry[name]


def describe_stage(name: str, stage: AgentReady | None = None) -> StageContract:
    """Return a stage's contract.

    With ``stage`` (an instance) -> dynamic contract with resolved key values.
    Otherwise -> instance-free ``static_contract`` (no resolved keys/cardinality).
    """
    if stage is not None:
        return build_contract(stage)
    return static_contract(get_agent_ready_stage_class(name))


def audio_stage_catalog(*, include_dynamic_defaults: bool = False) -> list[dict[str, Any]]:
    """Return the catalog as a list of ``{name, contract[, default_contract]}`` dicts.

    ``contract`` is the static (instance-free) contract. ``default_contract``
    (only with ``include_dynamic_defaults``) is the dynamic contract of a
    no-arg instance, attempted best-effort and ``None`` when the stage needs
    required constructor args.
    """
    entries: list[dict[str, Any]] = []
    for name in list_agent_ready_stages():
        cls = get_agent_ready_stage_class(name)
        entry: dict[str, Any] = {"name": name, "contract": static_contract(cls).to_dict()}
        if include_dynamic_defaults:
            try:
                entry["default_contract"] = build_contract(cls()).to_dict()
            except Exception:  # noqa: BLE001 - required-arg stages have no no-arg default
                entry["default_contract"] = None
        entries.append(entry)
    return entries


def catalog_as_json(*, include_dynamic_defaults: bool = False, indent: int | None = None) -> str:
    """JSON-serialized :func:`audio_stage_catalog` (an agent/UI tool schema)."""
    return json.dumps(audio_stage_catalog(include_dynamic_defaults=include_dynamic_defaults), indent=indent)
