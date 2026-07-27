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

"""Deterministic safety guardrails enforced INSIDE the verbs (not just the skill).

A weaker host, or a direct verb/CLI call, cannot bypass these:

* Workspace path lock  - file paths must resolve under an allowed root (opt-in via
  ``AUDIO_AGENT_WORKSPACE``); blocks traversal / reads-writes outside it.
* Secrets/transcript redaction - secret-looking keys and transcript text are
  stripped from verb return values before they reach the host LLM.
* Require-smoke evidence - a ``run`` can be made to refuse unless handed a valid
  ``smoke_token`` (opt-in via ``AUDIO_AGENT_REQUIRE_SMOKE``), proving a smoke ran
  for this exact recipe.

Semantic misuse refusal (e.g. "isolate a named person's voice") is NOT here: it
needs judgment a deterministic tool can't make, so it stays a skill/policy concern.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from functools import lru_cache
from typing import Any

_SMOKE_SECRET_ENV = "AUDIO_AGENT_SMOKE_SECRET"

# Substrings that mark a dict key as holding a secret (redacted from returns).
_SECRET_HINTS = ("token", "api_key", "apikey", "secret", "password", "aws_access_key", "credential")
# Keys holding transcript text (stripped from returns so transcripts don't reach the LLM).
_TRANSCRIPT_KEYS = frozenset({"text", "pred_text", "reference_text", "transcript", "text_ref"})
# Recipe param names that name a filesystem path (subject to the workspace lock).
_PATH_PARAM_HINTS = ("path", "dir", "manifest", "file_paths", "raw_data_dir")


def workspace_root() -> str | None:
    """The allowed workspace root, or None when the lock is not configured (default)."""
    root = os.environ.get("AUDIO_AGENT_WORKSPACE")
    return os.path.realpath(os.path.expanduser(root)) if root else None


def path_violations(paths: list[str | None]) -> list[str]:
    """Human-readable violations for any path outside the workspace.

    No-op (returns ``[]``) unless ``AUDIO_AGENT_WORKSPACE`` is set, so normal use
    with data outside the CWD is not blocked; locked-down deployments opt in.
    """
    root = workspace_root()
    if not root:
        return []
    out: list[str] = []
    for p in paths:
        if not p or not isinstance(p, str) or "://" in p:  # skip empty + remote URIs
            continue
        rp = os.path.realpath(os.path.expanduser(p))
        if rp != root and not rp.startswith(root + os.sep):
            out.append(f"{p!r} resolves outside the allowed workspace {root!r}")
    return out


def recipe_path_params(recipe: Any) -> list[str]:  # noqa: ANN401
    """Path-like string params across a recipe's stages (for the workspace lock)."""
    paths: list[str] = []
    for s in getattr(recipe, "stages", []) or []:
        for k, v in (getattr(s, "params", {}) or {}).items():
            if isinstance(v, str) and v and any(h in str(k).lower() for h in _PATH_PARAM_HINTS):
                paths.append(v)
    return paths


def redact(obj: Any, *, redact_transcripts: bool = True) -> Any:  # noqa: ANN401
    """Recursively strip secret-keyed values and (optionally) transcript text.

    Applied to verb return values so tokens never leak and transcripts don't enter
    the host LLM's context. Full transcripts remain in the output files on disk.
    """

    def _is_secret(k: Any) -> bool:  # noqa: ANN401
        lk = str(k).lower()
        return any(h in lk for h in _SECRET_HINTS)

    def _r(o: Any) -> Any:  # noqa: ANN401
        if isinstance(o, dict):
            out: dict[str, Any] = {}
            for k, v in o.items():
                if _is_secret(k):
                    out[k] = "<redacted-secret>"
                elif redact_transcripts and str(k).lower() in _TRANSCRIPT_KEYS and isinstance(v, str):
                    out[k] = f"<redacted-transcript:{len(v)}chars>"
                else:
                    out[k] = _r(v)
            return out
        if isinstance(o, list):
            return [_r(v) for v in o]
        return o

    return _r(obj)


@lru_cache(maxsize=1)
def _process_smoke_secret() -> bytes:
    """Random per-process secret (fallback when no stable secret is available)."""
    return secrets.token_bytes(32)


@lru_cache(maxsize=1)
def _smoke_secret() -> bytes:
    """HMAC key for smoke tokens.

    Precedence: ``AUDIO_AGENT_SMOKE_SECRET`` env (pin across machines / CI) > a
    per-deployment secret persisted once under the user cache (so a smoke in one
    process and the run in another share it) > a random per-process secret (still
    unforgeable within this process, e.g. the long-lived MCP server).
    """
    env = os.environ.get(_SMOKE_SECRET_ENV)
    if env:
        return env.encode("utf-8")
    cache = os.path.join(os.path.expanduser("~/.cache/nemo_curator"), "audio_agent_smoke.secret")
    try:
        if os.path.isfile(cache):
            with open(cache, "rb") as f:
                data = f.read().strip()
            if data:
                return data
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        fd = os.open(cache, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            secret = secrets.token_hex(32).encode("utf-8")
            f.write(secret)
        return secret
    except OSError:
        return _process_smoke_secret()


def smoke_token(config_hash: str | None) -> str:
    """Unforgeable proof that a smoke ran for this exact (frozen) recipe.

    HMAC over the config_hash keyed by a deployment/process secret (see
    :func:`_smoke_secret`), so — unlike a plain hash of the public config_hash — it
    cannot be minted by anything that merely knows the config_hash.
    """
    mac = hmac.new(_smoke_secret(), f"audio_agent_smoke|{config_hash}".encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()[:24]


def verify_smoke_token(token: str | None, config_hash: str | None) -> bool:
    """True iff ``token`` is the smoke token for ``config_hash`` (constant-time)."""
    return bool(token) and bool(config_hash) and hmac.compare_digest(token, smoke_token(config_hash))


def require_smoke() -> bool:
    """Whether ``run`` must refuse without a valid smoke token (opt-in)."""
    return os.environ.get("AUDIO_AGENT_REQUIRE_SMOKE", "").strip().lower() in ("1", "true", "yes", "on")
