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
"""Tests for the content-addressable per-stage cache."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from nemo_curator.agentic.cache import StageCache
from nemo_curator.agentic.ir import stage


class TestKeyDerivation:
    def test_same_inputs_same_key(self) -> None:
        s = stage("VADSegmentationStage", threshold=0.5)
        k1 = StageCache.derive_key(s, "upstreamA")
        k2 = StageCache.derive_key(s, "upstreamA")
        assert k1 == k2

    def test_param_change_changes_key(self) -> None:
        s1 = stage("VADSegmentationStage", threshold=0.5)
        s2 = stage("VADSegmentationStage", threshold=0.6)
        assert StageCache.derive_key(s1, "x") != StageCache.derive_key(s2, "x")

    def test_upstream_fingerprint_changes_key(self) -> None:
        s = stage("VADSegmentationStage")
        assert StageCache.derive_key(s, "A") != StageCache.derive_key(s, "B")


class TestRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path)
        key = "abc123"
        payload = {"hello": "world", "n": 42, "list": [1, 2, 3]}
        cache.save(key, "VADSegmentationStage", payload)
        loaded = cache.load(key)
        assert loaded == payload

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path)
        with pytest.raises(KeyError):
            cache.load("nope")

    def test_lookup_returns_none_for_missing(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path)
        assert cache.lookup("nope") is None


class TestGC:
    def test_gc_evicts_oldest_first(self, tmp_path: Path) -> None:
        # max_bytes sized so exactly one ~1 KB entry can survive after GC.
        cache = StageCache(tmp_path, max_bytes=1500)
        cache.save("first", "A", "x" * 1000)
        time.sleep(0.01)
        cache.save("second", "B", "y" * 1000)
        # Total would be ~2 KB > 1.5 KB → GC evicts the older entry.
        assert cache.lookup("first") is None
        assert cache.lookup("second") is not None

    def test_clear_wipes_everything(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path)
        cache.save("k", "S", {"a": 1})
        cache.clear()
        assert cache.lookup("k") is None
        assert cache.total_bytes() == 0

    def test_list_entries_reflects_saves(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path)
        cache.save("k1", "S1", "v1")
        cache.save("k2", "S2", "v2")
        names = {e.stage_name for e in cache.list_entries()}
        assert names == {"S1", "S2"}
