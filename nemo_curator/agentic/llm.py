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

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from loguru import logger


# ----------------------------------------------------------------------------
# Tier registry
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMTier:
    """Name → model mapping for the two-tier LLM strategy.

    Two roles, two model classes:

    - ``synth`` (the producers — Intent+Profile, Module Selection, Pipeline
      Staging, Parameter Tuning) needs the strongest instruction following
      and JSON discipline. Defaults to **Kimi K2** (1T-param MoE, ~32B
      active), which is currently the best open model at structured tool
      calling and stays fast thanks to MoE.
    - ``planner`` (the four critics) only emits a fixed checklist
      ``PASS``/``FIX`` verdict, so we use the much cheaper **Qwen3-30B-A3B**
      (3B active params).

    Reasoning models (DeepSeek-R1, etc.) are deliberately avoided here —
    their long chain-of-thought breaks ``response_format=json_object`` and
    inflates latency 3–5×.

    Override either via env vars ``CURATOR_ADV_PLANNER_MODEL`` /
    ``CURATOR_ADV_SYNTH_MODEL``.
    """

    planner: str = "qwen/qwen3-next-80b-a3b-instruct"
    synth: str = "qwen/qwen3-next-80b-a3b-instruct"


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
    surface without installing it. Both sync (``chat`` / ``chat_json``) and
    async (``achat`` / ``achat_json``) variants are exposed. The async pair
    is used by the team-based planner to fan out independent calls via
    :func:`asyncio.gather`; the sync pair is preserved for the existing
    ``dag`` and ``react`` planners.
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
    # Per-tier timeouts. The premium synth model is slow on first-call /
    # cold-load, so we default it to 4 minutes; the cheap planner model
    # stays at 60s. Override either with the matching env var.
    planner_timeout: float = field(
        default_factory=lambda: float(os.environ.get("CURATOR_ADV_PLANNER_TIMEOUT", "60")),
    )
    synth_timeout: float = field(
        default_factory=lambda: float(os.environ.get("CURATOR_ADV_SYNTH_TIMEOUT", "240")),
    )
    # Backwards-compat: any caller passing ``timeout=`` overrides BOTH tiers.
    timeout: float | None = None
    # Retry once on transient timeout / connection errors. The retry uses
    # exponential backoff (2s, 4s) with jitter. Set to 0 to disable.
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("CURATOR_ADV_LLM_RETRIES", "2")),
    )

    _client: Any = None  # lazy openai.OpenAI instance
    _async_client: Any = None  # lazy openai.AsyncOpenAI instance

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
        kwargs = self._build_kwargs(messages, model, tier, temperature, max_tokens, response_format)
        logger.debug(
            f"llm.chat(tier={tier}, model={model}, msgs={len(messages)}, "
            f"timeout={kwargs['timeout']}s)"
        )
        resp = self._call_with_retry(lambda: client.chat.completions.create(**kwargs), tier=tier)
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

    async def achat(
        self,
        messages: list[Message],
        *,
        tier: str = "planner",
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = 4096,
    ) -> str:
        """Asynchronous chat completion. Same contract as :meth:`chat`.

        Backed by ``openai.AsyncOpenAI`` so multiple in-flight requests can
        overlap. Use this from inside ``async def`` agents that fan out
        with :func:`asyncio.gather`.
        """

        client = self._ensure_async_client()
        model = self._model_for(tier)
        kwargs = self._build_kwargs(messages, model, tier, temperature, max_tokens, response_format)
        logger.debug(
            f"llm.achat(tier={tier}, model={model}, msgs={len(messages)}, "
            f"timeout={kwargs['timeout']}s)"
        )
        resp = await self._acall_with_retry(
            lambda: client.chat.completions.create(**kwargs), tier=tier,
        )
        return resp.choices[0].message.content or ""

    async def achat_json(
        self,
        messages: list[Message],
        *,
        tier: str = "planner",
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
    ) -> Any:
        """Async counterpart of :meth:`chat_json`."""

        text = await self.achat(
            messages,
            tier=tier,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        return _parse_json_lenient(text)

    # ---- Internals ------------------------------------------------------

    def _timeout_for(self, tier: str) -> float:
        if self.timeout is not None:
            return self.timeout
        if tier == "planner":
            return self.planner_timeout
        if tier == "synth":
            return self.synth_timeout
        return self.planner_timeout

    def _build_kwargs(
        self,
        messages: list[Message],
        model: str,
        tier: str,
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "timeout": self._timeout_for(tier),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def _call_with_retry(self, fn: Callable[[], Any], *, tier: str) -> Any:
        """Run a synchronous OpenAI call with exponential-backoff retries.

        Retries only on **transient** failures (timeouts, rate limits,
        5xx errors). Auth and request-shape errors are raised immediately
        so the caller doesn't waste budget on a hopeless retry.
        """

        attempts = max(1, self.max_retries + 1)
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if not _is_transient(exc) or i == attempts - 1:
                    raise
                last_exc = exc
                delay = (2 ** i) + random.uniform(0, 0.5)
                logger.warning(
                    f"llm.chat (tier={tier}) transient error {type(exc).__name__}: {exc!r}; "
                    f"retry {i + 1}/{attempts - 1} in {delay:.1f}s"
                )
                time.sleep(delay)
        assert last_exc is not None  # unreachable; for type-checkers
        raise last_exc

    async def _acall_with_retry(self, fn: Callable[[], Any], *, tier: str) -> Any:
        """Async counterpart of :meth:`_call_with_retry`."""

        attempts = max(1, self.max_retries + 1)
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001
                if not _is_transient(exc) or i == attempts - 1:
                    raise
                last_exc = exc
                delay = (2 ** i) + random.uniform(0, 0.5)
                logger.warning(
                    f"llm.achat (tier={tier}) transient error {type(exc).__name__}: {exc!r}; "
                    f"retry {i + 1}/{attempts - 1} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

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

    def _ensure_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - openai is in the venv
            msg = "openai SDK is required for LLMClient; install nemo_curator[agentic]."
            raise RuntimeError(msg) from exc
        self._async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._async_client

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
    and the tier and returns the assistant's content string.

    Optional ``async_delay`` (seconds) is honored by :meth:`achat` so tests
    can assert that fan-out actually overlapped (wall-clock <<
    N x async_delay)."""

    responder: Callable[[list[Message], str], str] = field(
        default=lambda _msgs, _tier: "{}",
    )
    async_delay: float = 0.0

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

    async def achat(
        self,
        messages: list[Message],
        *,
        tier: str = "planner",
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = 4096,
    ) -> str:
        if self.async_delay > 0:
            import asyncio  # noqa: PLC0415
            await asyncio.sleep(self.async_delay)
        return self.responder(messages, tier)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    """True if ``exc`` is a class of error that's worth retrying.

    We intentionally do **not** import openai / httpx eagerly so this
    module stays importable in test environments without those SDKs.
    Detection is duck-typed on class name + status code where available.
    """

    name = type(exc).__name__
    # OpenAI SDK v1.x raises these names; httpx raises Timeout/Connect*.
    if name in {
        "APITimeoutError", "APIConnectionError",
        "TimeoutException", "ReadTimeout", "WriteTimeout", "ConnectTimeout",
        "ConnectError", "RemoteProtocolError",
        "RateLimitError", "InternalServerError",
        "APIError",  # generic upstream-side hiccup
    }:
        return True
    # Some SDKs expose a numeric ``status_code`` attribute on the error.
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in {408, 409, 425, 429} or (isinstance(status, int) and 500 <= status < 600):
        return True
    return False


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
