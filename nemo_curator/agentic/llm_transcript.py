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
"""Per-request LLM call transcript.

A tiny, dependency-free recorder that captures every chat-completion the
``LLMClient`` makes during one HTTP request. The web app binds it for
the duration of ``plan()`` / ``build()``; each call's request messages,
response text, latency, retries, and errors are appended as one JSONL
line to a per-run log.

Design choices:

- ``contextvars`` are used so concurrent requests (and async agents) get
  separate log targets without threading bookkeeping through every
  function signature.
- Recording is **opt-in**: when no transcript is bound the recorder is a
  no-op, so library callers (tests, the CLI) pay nothing.
- Each record is a flat dict that fits one JSONL line — no nesting that
  ``jq`` users have to dig through.
- Long message bodies are truncated to ``CURATOR_ADV_LLM_LOG_MAX_CHARS``
  per field (default 4096) so a chatty session prompt doesn't bloat the
  log; the original lengths are kept under ``..._chars``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


_DEFAULT_MAX_CHARS = int(os.environ.get("CURATOR_ADV_LLM_LOG_MAX_CHARS", "4096"))


@dataclass
class LLMTranscript:
    """One log target. Append-only JSONL on disk plus an in-memory tail."""

    path: Path
    run_id: str | None = None
    user_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    max_chars: int = _DEFAULT_MAX_CHARS
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _records: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _truncate(self, s: str) -> tuple[str, int, bool]:
        if not isinstance(s, str):
            s = str(s)
        n = len(s)
        if n <= self.max_chars:
            return s, n, False
        return s[: self.max_chars] + f"…[truncated {n - self.max_chars} chars]", n, True

    def record(self, payload: dict[str, Any]) -> None:
        """Append one JSON record. Never raises — logging must not break callers."""

        try:
            row = {
                "ts": payload.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": self.run_id,
                "user_id": self.user_id,
                **self.tags,
                **payload,
            }
            with self._lock:
                self._records.append(row)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001
            # Never fail the LLM call because the recorder broke. Best
            # effort: write to stderr via the loguru logger if available.
            try:
                from loguru import logger  # noqa: PLC0415

                logger.warning(f"llm_transcript: failed to record call: {exc}")
            except Exception:  # noqa: BLE001
                pass

    def messages(self) -> list[dict[str, Any]]:
        """Snapshot of in-memory records (does not re-read the file)."""

        with self._lock:
            return list(self._records)


_active: ContextVar[LLMTranscript | None] = ContextVar("llm_transcript_active", default=None)


def current_transcript() -> LLMTranscript | None:
    return _active.get()


@contextmanager
def bind_transcript(transcript: LLMTranscript | None) -> Iterator[LLMTranscript | None]:
    """Bind *transcript* for the duration of the ``with`` block.

    Passing ``None`` is allowed and disables recording inside the block
    (useful for tests that want to silence an outer binding).
    """

    token = _active.set(transcript)
    try:
        yield transcript
    finally:
        _active.reset(token)


@dataclass
class _PendingCall:
    """Builder that holds the call request until the response arrives."""

    transcript: LLMTranscript
    call_id: str
    started_at: float
    record: dict[str, Any]

    def finish(
        self,
        *,
        response_text: str | None = None,
        error: str | None = None,
        retries: int = 0,
    ) -> None:
        latency = time.monotonic() - self.started_at
        truncated, _, _ = self.transcript._truncate(response_text or "")
        rec = {**self.record}
        rec.update({
            "response_text": truncated,
            "response_chars": len(response_text or ""),
            "latency_s": round(latency, 4),
            "retries": retries,
            "error": error,
            "status": "ok" if error is None else "error",
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self.transcript.record(rec)


def start_call(
    *,
    purpose: str,
    tier: str,
    model: str,
    base_url: str | None,
    messages: list[dict[str, str]] | None,
    temperature: float,
    max_tokens: int | None,
    response_format: dict[str, Any] | None,
) -> _PendingCall | None:
    """Begin recording one LLM call.

    Returns ``None`` when no transcript is bound, so the caller can
    skip ``.finish()`` cheaply.
    """

    transcript = _active.get()
    if transcript is None:
        return None
    call_id = uuid.uuid4().hex[:12]
    started_at = time.monotonic()
    msgs_clean: list[dict[str, Any]] = []
    for m in messages or []:
        role = m.get("role", "")
        content = m.get("content", "")
        truncated, n_chars, was_truncated = transcript._truncate(content)
        msgs_clean.append({
            "role": role,
            "content": truncated,
            "chars": n_chars,
            "truncated": was_truncated,
        })
    record = {
        "call_id": call_id,
        "purpose": purpose,
        "tier": tier,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": (response_format or {}).get("type"),
        "messages": msgs_clean,
        "n_messages": len(msgs_clean),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return _PendingCall(
        transcript=transcript,
        call_id=call_id,
        started_at=started_at,
        record=record,
    )


__all__ = [
    "LLMTranscript",
    "bind_transcript",
    "current_transcript",
    "start_call",
]
