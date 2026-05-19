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
"""Content-addressable per-stage cache.

Stage outputs are checkpointed by the runner into
``<target_dir>/.adv/cache/<stage_key>.json`` and a sidecar
``<stage_key>.payload`` for the pickled task list. The cache key is a hash of
``(stage_name, stage_params, upstream_fingerprint)`` so a different upstream
necessarily invalidates the entry. LRU GC keeps the cache under a configurable
byte budget.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from nemo_curator.agentic.ir import StageRef


@dataclass
class CacheEntry:
    """One on-disk cache record."""

    key: str
    stage_name: str
    payload_path: Path
    meta_path: Path
    byte_size: int
    saved_at: float


class StageCache:
    """File-system cache for stage outputs."""

    def __init__(self, root: str | Path, *, max_bytes: int = 50 * 1024 * 1024 * 1024) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)

    # ---- Key derivation ---------------------------------------------------

    @staticmethod
    def derive_key(stage_ref: StageRef, upstream_fingerprint: str) -> str:
        h = hashlib.sha256()
        h.update(stage_ref.stage.encode())
        h.update(b"\x00")
        h.update(json.dumps(stage_ref.params, sort_keys=True, default=str).encode())
        h.update(b"\x00")
        h.update(upstream_fingerprint.encode())
        return h.hexdigest()[:24]

    # ---- IO --------------------------------------------------------------

    def lookup(self, key: str) -> CacheEntry | None:
        payload = self.root / f"{key}.payload"
        meta = self.root / f"{key}.meta.json"
        if not payload.exists() or not meta.exists():
            return None
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            return CacheEntry(
                key=key,
                stage_name=info["stage_name"],
                payload_path=payload,
                meta_path=meta,
                byte_size=payload.stat().st_size,
                saved_at=info["saved_at"],
            )
        except Exception:  # noqa: BLE001
            return None

    def load(self, key: str) -> Any:
        entry = self.lookup(key)
        if entry is None:
            msg = f"cache miss for {key!r}"
            raise KeyError(msg)
        with entry.payload_path.open("rb") as f:
            return pickle.load(f)

    def save(self, key: str, stage_name: str, value: Any) -> CacheEntry:
        payload = self.root / f"{key}.payload"
        meta = self.root / f"{key}.meta.json"
        with payload.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        info = {"stage_name": stage_name, "saved_at": time.time()}
        meta.write_text(json.dumps(info), encoding="utf-8")
        size = payload.stat().st_size
        logger.info(f"cache: wrote {stage_name} → {key} ({size / 1e6:.1f} MB)")
        self._maybe_gc()
        return CacheEntry(
            key=key,
            stage_name=stage_name,
            payload_path=payload,
            meta_path=meta,
            byte_size=size,
            saved_at=info["saved_at"],
        )

    def list_entries(self) -> list[CacheEntry]:
        out: list[CacheEntry] = []
        for meta in self.root.glob("*.meta.json"):
            key = meta.stem.removesuffix(".meta")
            entry = self.lookup(key)
            if entry is not None:
                out.append(entry)
        return out

    def total_bytes(self) -> int:
        return sum(e.byte_size for e in self.list_entries())

    def gc(self, target_bytes: int | None = None) -> int:
        """Evict oldest entries until under ``target_bytes`` (default ``max_bytes``)."""

        target = target_bytes if target_bytes is not None else self.max_bytes
        entries = sorted(self.list_entries(), key=lambda e: e.saved_at)
        freed = 0
        while entries and self.total_bytes() > target:
            victim = entries.pop(0)
            try:
                victim.payload_path.unlink()
                victim.meta_path.unlink()
            except FileNotFoundError:
                continue
            freed += victim.byte_size
            logger.info(f"cache: gc evicted {victim.stage_name} ({victim.key})")
        return freed

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- Internals -------------------------------------------------------

    def _maybe_gc(self) -> None:
        if self.total_bytes() > self.max_bytes:
            self.gc()


__all__ = ["CacheEntry", "StageCache"]
