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
"""CapabilityRegistry — load and index stage cards.

Card discovery covers three locations (in priority order):

1. **In-tree cards** — ``stage_card.yaml`` files co-located with stage modules
   under ``nemo_curator/stages/`` (or any subdirectory thereof).
2. **User-local cards** — under ``~/.curator-adv/plugins/<name>/stage_card.yaml``
   (where the wizard installs onboarded third-party stages).
3. **Project-local cards** — under ``./.curator-adv/plugins/<name>/stage_card.yaml``
   (CWD-scoped, takes precedence over user-local for development).

Every loaded card is cross-checked against the in-process
``nemo_curator.stages.base._STAGE_REGISTRY`` so the agent never plans a
``StageCard`` whose Python class is unimportable or is not actually a
:class:`ProcessingStage`. Missing-import cards land in the registry's
``unresolved`` bucket so ``curator-adv list-stages`` can flag them.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from loguru import logger

from nemo_curator.agentic.cards import CapabilityTag, StageCard, load_stage_card

# Filename convention everywhere on disk
STAGE_CARD_FILENAME = "stage_card.yaml"


@dataclass(frozen=True)
class UnresolvedCard:
    """A card that loaded but whose ``target`` cannot be imported.

    Surfaced by the registry so the user sees a clean error rather than a
    silently-missing capability.
    """

    card: StageCard
    error: str
    source_path: Path


class RegistryEntry:
    """One record in the registry.

    The class is resolved lazily via :attr:`klass` — the first access pays the
    import cost, every subsequent call hits the cached attribute. This keeps
    cards-only operations (``list-stages``, ``inspect``, ``plan``, ``validate``)
    independent of the heavy NeMo/torch import chain.
    """

    __slots__ = ("card", "source_path", "origin", "_klass", "_resolve_error")

    def __init__(
        self,
        card: StageCard,
        source_path: Path,
        origin: str,
        klass: type | None = None,
    ) -> None:
        self.card = card
        self.source_path = source_path
        self.origin = origin
        self._klass: type | None = klass
        self._resolve_error: str | None = None

    @property
    def klass(self) -> type:
        if self._klass is not None:
            return self._klass
        if self._resolve_error is not None:
            msg = f"target {self.card.target!r} could not be imported: {self._resolve_error}"
            raise ImportError(msg)
        klass = _resolve_class(self.card.target)
        if klass is None:
            self._resolve_error = "import failed"
            msg = f"target {self.card.target!r} could not be imported"
            raise ImportError(msg)
        if not _is_processing_stage(klass):
            self._resolve_error = "not a ProcessingStage subclass"
            msg = f"target {self.card.target!r} is not a ProcessingStage subclass"
            raise TypeError(msg)
        self._klass = klass
        return klass

    def try_klass(self) -> type | None:
        try:
            return self.klass
        except (ImportError, TypeError):
            return None


@dataclass
class CapabilityRegistry:
    """Index of loaded stage cards keyed by class name."""

    by_name: dict[str, RegistryEntry] = field(default_factory=dict)
    unresolved: list[UnresolvedCard] = field(default_factory=list)
    duplicate_warnings: list[str] = field(default_factory=list)

    # ---- Public lookup API -------------------------------------------

    def get(self, name: str) -> RegistryEntry | None:
        return self.by_name.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.by_name

    def all_cards(self) -> list[StageCard]:
        return [e.card for e in self.by_name.values()]

    def search_by_capability(
        self,
        tag: CapabilityTag,
        *,
        include_also_handles: bool = True,
        commercial_only: bool = False,
    ) -> list[RegistryEntry]:
        """Return entries whose ``capabilities`` (and optionally ``also_handles``) include ``tag``."""

        out: list[RegistryEntry] = []
        for e in self.by_name.values():
            tags = set(e.card.capabilities)
            if include_also_handles:
                tags |= set(e.card.also_handles)
            if tag in tags:
                if commercial_only and not e.card.commercial_safe:
                    continue
                out.append(e)
        return out

    def search_by_text(self, query: str) -> list[RegistryEntry]:
        """Naïve substring match over name + summary + tags. Phase 1 placeholder."""

        q = query.lower()
        out: list[RegistryEntry] = []
        for e in self.by_name.values():
            hay = " ".join([
                e.card.name,
                e.card.summary,
                e.card.description,
                " ".join(e.card.tags),
                " ".join(t.value for t in e.card.capabilities),
            ]).lower()
            if q in hay:
                out.append(e)
        return out


# ----------------------------------------------------------------------------
# Discovery helpers
# ----------------------------------------------------------------------------


def _in_tree_roots() -> list[Path]:
    """Where in-tree cards live.

    Defaults to ``<nemo_curator package>/stages``. Override via env var
    ``CURATOR_ADV_IN_TREE_ROOTS`` (path-list, colon-separated).
    """

    env = os.environ.get("CURATOR_ADV_IN_TREE_ROOTS")
    if env:
        return [Path(p).expanduser().resolve() for p in env.split(os.pathsep) if p]

    import nemo_curator.stages as stages_pkg

    return [Path(stages_pkg.__file__).parent.resolve()]


def _user_root() -> Path:
    return Path(os.environ.get("CURATOR_ADV_USER_PLUGINS", "~/.curator-adv/plugins")).expanduser().resolve()


def _project_root() -> Path:
    cwd = Path(os.environ.get("CURATOR_ADV_PROJECT_PLUGINS", str(Path.cwd() / ".curator-adv" / "plugins"))).expanduser()
    return cwd.resolve()


def _discover_cards(roots: Iterable[Path]) -> list[Path]:
    """Walk a set of roots and return every ``stage_card.yaml`` under them."""

    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob(STAGE_CARD_FILENAME):
            if p.is_file():
                found.append(p.resolve())
    return sorted(set(found))


# ----------------------------------------------------------------------------
# Load + cross-check
# ----------------------------------------------------------------------------


def _resolve_class(target: str) -> type | None:
    """Import ``target`` (``module.path.ClassName``) and return the class, or ``None``."""

    if ":" in target or "." not in target:
        return None
    module_path, _, class_name = target.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=False).debug(f"target import failed for {target}: {exc}")
        return None
    klass = getattr(module, class_name, None)
    if klass is None or not isinstance(klass, type):
        return None
    return klass


def _is_processing_stage(klass: type) -> bool:
    """True if ``klass`` is a subclass of ``ProcessingStage`` (without importing it eagerly).

    Inspecting MRO avoids a hard import dependency from this module on the
    base package — although by the time we're here, ``ProcessingStage`` is
    already imported anyway via ``_STAGE_REGISTRY``.
    """

    return any(b.__name__ == "ProcessingStage" for b in klass.__mro__)


def build_registry(
    *,
    extra_roots: Iterable[Path] | None = None,
    strict: bool = False,
    cross_check_runtime: bool = False,
    eager: bool = False,
) -> CapabilityRegistry:
    """Discover and load every stage card. Class resolution is lazy by default.

    Parameters
    ----------
    extra_roots
        Extra plugin roots to scan (used by tests and the wizard).
    strict
        If ``True``, raise on any unresolvable card. Only meaningful when
        ``eager=True`` (otherwise import errors surface on first
        ``entry.klass`` access).
    cross_check_runtime
        Look up each card's name in
        ``nemo_curator.stages.base._STAGE_REGISTRY`` to confirm self-
        registration. Triggers the full NeMo import chain — only enable when
        you actually need the runtime cross-check (``lint``).
    eager
        Resolve every card's target at build time. Off by default so
        ``list-stages`` / ``inspect`` / ``plan`` / ``validate`` are fast.
    """

    registry = CapabilityRegistry()

    _stage_registry: dict | None = None
    if cross_check_runtime:
        from nemo_curator.stages.base import _STAGE_REGISTRY  # noqa: PLC0415

        _stage_registry = _STAGE_REGISTRY

    in_tree = _in_tree_roots()
    user_local = [_user_root()]
    project_local = [_project_root()]
    extras = [Path(p).expanduser().resolve() for p in (extra_roots or [])]

    bundles = [
        ("in_tree", in_tree + extras),
        ("user_local", user_local),
        ("project_local", project_local),
    ]

    for origin, roots in bundles:
        for card_path in _discover_cards(roots):
            try:
                card = load_stage_card(card_path)
            except Exception as exc:  # noqa: BLE001
                msg = f"Could not parse stage card {card_path}: {exc}"
                if strict:
                    raise RuntimeError(msg) from exc
                logger.warning(msg)
                continue

            klass: type | None = None
            if eager:
                klass = _resolve_class(card.target)
                if klass is None:
                    registry.unresolved.append(UnresolvedCard(
                        card=card,
                        error=f"target {card.target!r} not importable",
                        source_path=card_path,
                    ))
                    if strict:
                        msg = f"Could not import {card.target!r} for card {card_path}"
                        raise RuntimeError(msg)
                    continue

                if not _is_processing_stage(klass):
                    registry.unresolved.append(UnresolvedCard(
                        card=card,
                        error=f"target {card.target!r} is not a ProcessingStage subclass",
                        source_path=card_path,
                    ))
                    if strict:
                        msg = f"target {card.target!r} is not a ProcessingStage"
                        raise RuntimeError(msg)
                    continue

                if _stage_registry is not None and card.name not in _stage_registry:
                    logger.warning(
                        f"Card {card.name!r} loaded but not in _STAGE_REGISTRY; "
                        f"its class may be defined under a non-imported package."
                    )

            if card.name in registry.by_name:
                prev = registry.by_name[card.name]
                if prev.origin == origin:
                    registry.duplicate_warnings.append(
                        f"Duplicate card for {card.name!r} from {origin}: {card_path} (kept {prev.source_path})"
                    )
                    continue
                logger.info(
                    f"Card {card.name!r} from {origin} ({card_path}) overrides {prev.origin} ({prev.source_path})"
                )

            registry.by_name[card.name] = RegistryEntry(
                card=card,
                klass=klass,
                source_path=card_path,
                origin=origin,
            )

    return registry


# ----------------------------------------------------------------------------
# Drift detection (Step 1.6)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftFinding:
    """One drift between a card and its underlying Python class."""

    card_name: str
    kind: str  # "class_name_mismatch" | "missing_param" | "extra_param" | "registry_missing"
    detail: str


# Fields that are modeled on the card outside ``params`` (they live on
# top-level StageCard attributes like ``resources`` / ``batch_size``) or are
# infrastructure-only and should never appear in a user-facing param list.
_INFRASTRUCTURE_FIELDS: frozenset[str] = frozenset({
    "name",
    "resources",
    "batch_size",
    "runtime_env",
})


def _is_user_facing(param_name: str) -> bool:
    """Skip dunder + private (``_foo``) attributes; they are internal state."""

    if param_name.startswith("_"):
        return False
    return param_name not in _INFRASTRUCTURE_FIELDS


def diff_card_vs_class(card: StageCard, klass: type) -> list[DriftFinding]:
    """Return any drifts between ``card`` and its underlying ``ProcessingStage`` class.

    Only user-facing kwargs are compared. Infrastructure fields modeled
    elsewhere on the card (``resources``, ``batch_size``, ``name``,
    ``runtime_env``) and private attributes (``_foo``) are excluded.
    """

    import inspect

    findings: list[DriftFinding] = []

    if klass.__name__ != card.name:
        findings.append(DriftFinding(
            card_name=card.name,
            kind="class_name_mismatch",
            detail=f"card.name={card.name!r} but class.__name__={klass.__name__!r}",
        ))

    try:
        sig = inspect.signature(klass.__init__)
    except (TypeError, ValueError):
        return findings

    init_params = {
        n: p
        for n, p in sig.parameters.items()
        if n not in {"self", "args", "kwargs"}
        and p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
        and _is_user_facing(n)
    }
    card_params = {p.name for p in card.params}

    for n in init_params:
        if n not in card_params:
            findings.append(DriftFinding(
                card_name=card.name,
                kind="missing_param",
                detail=f"__init__ parameter {n!r} not documented in card.params",
            ))
    for n in card_params:
        if n not in init_params:
            findings.append(DriftFinding(
                card_name=card.name,
                kind="extra_param",
                detail=f"card.params lists {n!r} but __init__ has no such parameter",
            ))

    return findings


def lint(registry: CapabilityRegistry) -> list[DriftFinding]:
    """Run drift detection across every entry in the registry.

    Triggers class resolution for every card; expect a real cost the first
    time you call this in a process (the NeMo / torch / silero imports land
    here).
    """

    out: list[DriftFinding] = []
    for entry in registry.by_name.values():
        klass = entry.try_klass()
        if klass is None:
            out.append(DriftFinding(
                card_name=entry.card.name,
                kind="registry_missing",
                detail=f"target {entry.card.target!r} could not be imported",
            ))
            continue
        out.extend(diff_card_vs_class(entry.card, klass))
    return out


__all__ = [
    "CapabilityRegistry",
    "DriftFinding",
    "RegistryEntry",
    "STAGE_CARD_FILENAME",
    "UnresolvedCard",
    "build_registry",
    "diff_card_vs_class",
    "lint",
]
