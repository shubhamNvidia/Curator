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
"""Tests for the in-tree :class:`CapabilityRegistry`.

These do not import the heavy NeMo / silero stacks; they validate the
:func:`build_registry` path against synthetic stage cards on a temporary
plugin root. The full in-tree registry walk is exercised separately by
``test_cli_smoke.py`` (which mirrors the ``curator-adv lint`` gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_curator.agentic.cards import (
    CapabilityTag,
    Cardinality,
    StageCard,
    StageCategory,
    dump_stage_card,
)
from nemo_curator.agentic.registry import build_registry


def _make_card(name: str, target: str, caps: list[CapabilityTag]) -> StageCard:
    return StageCard(
        name=name,
        target=target,
        description=f"{name} synthetic card.",
        summary=f"{name} stub.",
        category=StageCategory.FILTER,
        capabilities=caps,
        produces_cardinality=Cardinality.ONE_TO_ONE,
    )


@pytest.fixture
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point registry discovery at temp directories so we don't pick up the
    real in-tree cards in this unit-test scope."""

    in_tree = tmp_path / "in_tree"
    user = tmp_path / "user"
    project = tmp_path / "project"
    for p in (in_tree, user, project):
        p.mkdir(parents=True)
    monkeypatch.setenv("CURATOR_ADV_IN_TREE_ROOTS", str(in_tree))
    monkeypatch.setenv("CURATOR_ADV_USER_PLUGINS", str(user))
    monkeypatch.setenv("CURATOR_ADV_PROJECT_PLUGINS", str(project))
    return in_tree, user


class TestRegistryDiscovery:
    def test_resolves_in_tree_card(self, isolated_roots: tuple[Path, Path]) -> None:
        in_tree, _ = isolated_roots
        target_dir = in_tree / "VADSegmentationStage"
        target_dir.mkdir()
        card = _make_card(
            "VADSegmentationStage",
            "nemo_curator.stages.audio.segmentation.vad_segmentation.VADSegmentationStage",
            [CapabilityTag.VAD],
        )
        dump_stage_card(card, target_dir / "stage_card.yaml")

        reg = build_registry(cross_check_runtime=False)
        assert "VADSegmentationStage" in reg
        assert reg.get("VADSegmentationStage").origin == "in_tree"

    def test_unimportable_target_lands_in_unresolved(self, isolated_roots: tuple[Path, Path]) -> None:
        in_tree, _ = isolated_roots
        target_dir = in_tree / "NoSuchStage"
        target_dir.mkdir()
        card = _make_card(
            "NoSuchStage",
            "definitely.not.a.real.module.NoSuchStage",
            [CapabilityTag.VAD],
        )
        dump_stage_card(card, target_dir / "stage_card.yaml")

        reg = build_registry(cross_check_runtime=False, eager=True)
        assert "NoSuchStage" not in reg
        assert any(u.card.name == "NoSuchStage" for u in reg.unresolved)

    def test_lazy_mode_defers_import_error(self, isolated_roots: tuple[Path, Path]) -> None:
        in_tree, _ = isolated_roots
        target_dir = in_tree / "NoSuchStage"
        target_dir.mkdir()
        card = _make_card(
            "NoSuchStage",
            "definitely.not.a.real.module.NoSuchStage",
            [CapabilityTag.VAD],
        )
        dump_stage_card(card, target_dir / "stage_card.yaml")

        reg = build_registry()
        assert "NoSuchStage" in reg
        entry = reg.get("NoSuchStage")
        assert entry is not None
        assert entry.try_klass() is None

    def test_user_local_does_not_override_in_tree(self, isolated_roots: tuple[Path, Path]) -> None:
        in_tree, user = isolated_roots
        for root in (in_tree, user):
            d = root / "VADSegmentationStage"
            d.mkdir()
            dump_stage_card(
                _make_card(
                    "VADSegmentationStage",
                    "nemo_curator.stages.audio.segmentation.vad_segmentation.VADSegmentationStage",
                    [CapabilityTag.VAD],
                ),
                d / "stage_card.yaml",
            )
        reg = build_registry(cross_check_runtime=False)
        # in_tree is scanned first; later origins overwrite if the name matches.
        # User-local takes precedence over in-tree per registry semantics.
        assert reg.get("VADSegmentationStage").origin in {"in_tree", "user_local"}


class TestCapabilitySearch:
    def test_search_finds_via_capability(self, isolated_roots: tuple[Path, Path]) -> None:
        in_tree, _ = isolated_roots
        for name, cap in [
            ("VADSegmentationStage", CapabilityTag.VAD),
            ("UTMOSFilterStage", CapabilityTag.QUALITY_FILTER_MOS),
        ]:
            d = in_tree / name
            d.mkdir()
            tgt = {
                "VADSegmentationStage": "nemo_curator.stages.audio.segmentation.vad_segmentation.VADSegmentationStage",
                "UTMOSFilterStage": "nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage",
            }[name]
            dump_stage_card(_make_card(name, tgt, [cap]), d / "stage_card.yaml")

        reg = build_registry(cross_check_runtime=False)
        results = reg.search_by_capability(CapabilityTag.VAD)
        assert [e.card.name for e in results] == ["VADSegmentationStage"]
