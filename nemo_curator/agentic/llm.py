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
"""Minimal LLM client used by the agentic core.

The client speaks the OpenAI Chat Completions API. NVIDIA NIM endpoints
(``https://integrate.api.nvidia.com/v1``) are wire-compatible, so the same
client works for both NIM and local OpenAI-style servers.

Design choices:

- The client never depends on NAT; NAT is layered on top later.
- Two named tiers are exposed via :data:`LLM_TIERS` so the planner and critic
  can pick cheap vs. premium without hard-coding model names.
- The :class:`MockLLM` subclass lets tests inject deterministic responses
  without ever opening a socket — used by ``test_plan_agent.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from loguru import logger


# ----------------------------------------------------------------------------
# Tier registry
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMTier:
    """Name → model mapping for the two-tier LLM strategy.

    ``planner`` is the cheap tier used for intent extraction and capability
    matching. ``synth`` is the premium tier used for IR synthesis and the
    LLM-driven critic. Override via env vars
    ``CURATOR_ADV_PLANNER_MODEL`` / ``CURATOR_ADV_SYNTH_MODEL``.
    """

    planner: str = "nvidia/nemotron-3-nano-30b-a3b"
    synth: str = "nvidia/nemotron-3-super-120b-a12b"


def default_tiers() -> LLMTier:
    return LLMTier(
        planner=os.environ.get("CURATOR_ADV_PLANNER_MODEL", LLMTier.planner),
        synth=os.environ.get("CURATOR_ADV_SYNTH_MODEL", LLMTier.synth),
    )


# ----------------------------------------------------------------------------
# Message shape
# ----------------------------------------------------------------------------


@dataclass
class Message:
    """One chat-completions message."""

    role: str
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def sys(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------


@dataclass
class LLMClient:
    """Thin OpenAI-compatible client.

    The ``openai`` SDK is imported lazily so tests can stub the whole call
    surface without installing it.
    """

    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "CURATOR_ADV_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1",
        ),
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("CURATOR_ADV_LLM_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    tiers: LLMTier = field(default_factory=default_tiers)
    timeout: float = 60.0

    _client: Any = None  # lazy openai.OpenAI instance

    # ---- Public API -----------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        *,
        tier: str = "planner",
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = 4096,
    ) -> str:
        """Synchronous chat completion. Returns the assistant message body."""

        client = self._ensure_client()
        model = self._model_for(tier)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        logger.debug(f"llm.chat(tier={tier}, model={model}, msgs={len(messages)})")

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[Message],
        *,
        tier: str = "planner",
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
    ) -> Any:
        """Helper that forces JSON output and parses it.

        Falls back to extracting the first ``{...}`` block if the model
        ignores ``response_format``.
        """

        text = self.chat(
            messages,
            tier=tier,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        return _parse_json_lenient(text)

    # ---- Internals ------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - openai is in the venv
            msg = "openai SDK is required for LLMClient; install nemo_curator[agentic]."
            raise RuntimeError(msg) from exc
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _model_for(self, tier: str) -> str:
        if tier == "planner":
            return self.tiers.planner
        if tier == "synth":
            return self.tiers.synth
        msg = f"Unknown LLM tier {tier!r}; expected 'planner' or 'synth'."
        raise ValueError(msg)


# ----------------------------------------------------------------------------
# Mock client for tests + offline planning
# ----------------------------------------------------------------------------


@dataclass
class MockLLM(LLMClient):
    """Test double. ``responder`` is a callable that receives the messages
    and the tier and returns the assistant's content string."""

    responder: Callable[[list[Message], str], str] = field(
        default=lambda _msgs, _tier: "{}",
    )

    def chat(
        self,
        messages: list[Message],
        *,
        tier: str = "planner",
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = 4096,
    ) -> str:
        return self.responder(messages, tier)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _parse_json_lenient(text: str) -> Any:
    """Extract the first JSON object from ``text``; raise on failure."""

    text = (text or "").strip()
    if not text:
        msg = "LLM returned an empty string."
        raise ValueError(msg)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: pull the substring between the first '{' and matching '}'.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        msg = f"LLM output is not valid JSON: {text[:200]!r}"
        raise ValueError(msg)
    return json.loads(text[start : end + 1])


__all__ = [
    "LLMClient",
    "LLMTier",
    "Message",
    "MockLLM",
    "assistant",
    "default_tiers",
    "sys",
    "user",
]
