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

import base64
import hashlib
import hmac
import os
import re
import secrets
from functools import lru_cache
from typing import Any
from urllib.parse import unquote, urlsplit

_SMOKE_SECRET_ENV = "AUDIO_AGENT_SMOKE_SECRET"
_LOCAL_URI_SCHEMES = frozenset({"file", "local"})

# Substrings that mark a dict key as holding a secret (redacted from returns).
_SECRET_HINTS = ("token", "api_key", "apikey", "secret", "password", "aws_access_key", "credential")
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?
        [a-z0-9_-]*
        (?:token|api[_-]?key|access[_-]?key|password|secret|credential)
        [a-z0-9_-]*
        ["']?
        \s*[:=]\s*
    )
    (?P<value>
        "(?:\\.|[^"\\])*"
        |
        '(?:\\.|[^'\\])*'
        |
        [^\s,;}\]]+
    )
    """
)
_HF_TOKEN_VALUE = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_BASIC_VALUE = re.compile(
    r"(?i)(?P<prefix>\bbasic\s+)(?P<value>[A-Za-z0-9+/]{4,}={0,2})(?=$|[^A-Za-z0-9+/=])"
)
_URL_USERINFO = re.compile(
    r"(?i)(?P<scheme>\b[a-z][a-z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@"
)
_JWT_VALUE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"(?![A-Za-z0-9_-])"
)
# Strong, well-known prefixes plus a minimum opaque suffix length keep this
# conservative: ordinary strings such as ``sk-learn`` are left untouched.
_PREFIXED_TOKEN_VALUE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|nvapi-[A-Za-z0-9_-]{16,}"
    r"|xox[baprs]-[A-Za-z0-9-]{16,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r"|AIza[A-Za-z0-9_-]{20,}"
    r")(?![A-Za-z0-9_-])"
)
# Keys holding transcript text (stripped from returns so transcripts don't reach the LLM).
_TRANSCRIPT_KEYS = frozenset({"text", "pred_text", "reference_text", "transcript", "text_ref"})
# Dataset and output parameters that the agent itself reads or writes. This is a
# closed list: substring matching mistakes semantic fields such as
# ``audio_filepath_key`` and ``audio_path_resolution`` for filesystem paths, and
# mistakes model IDs such as ``nvidia/...`` for workspace-relative files.
_PATH_PARAM_NAMES = frozenset(
    {
        "audio_dir",
        "data_dir",
        "file_paths",
        "input_manifest",
        "manifest",
        "manifest_path",
        "output_audio_tar_path",
        "output_dir",
        "output_manifest",
        "output_path",
        "raw_data_dir",
        "resampled_audio_dir",
        "rttm_out_dir",
        "separated_audio_dir",
    }
)


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
        if not p or not isinstance(p, str):
            continue
        parsed = urlsplit(p)
        if parsed.scheme and parsed.scheme not in _LOCAL_URI_SCHEMES:
            continue  # a local workspace root cannot constrain a remote namespace
        local = unquote(parsed.path) if parsed.scheme in _LOCAL_URI_SCHEMES else p
        if parsed.scheme in _LOCAL_URI_SCHEMES and parsed.netloc not in ("", "localhost"):
            local = f"//{parsed.netloc}{local}"
        rp = os.path.realpath(os.path.expanduser(local))
        if rp != root and not rp.startswith(root + os.sep):
            out.append(f"{p!r} resolves outside the allowed workspace {root!r}")
    return out


def recipe_path_params(recipe: Any) -> list[str]:  # noqa: ANN401
    """Path-like string params across a recipe's stages (for the workspace lock)."""
    paths: list[str] = []
    for s in getattr(recipe, "stages", []) or []:
        for k, v in (getattr(s, "params", {}) or {}).items():
            if str(k).lower() not in _PATH_PARAM_NAMES:
                continue
            values = v if isinstance(v, (list, tuple)) else [v]
            paths.extend(item for item in values if isinstance(item, str) and item)
    return paths


def redact_secret_text(value: str) -> str:
    """Redact credential values embedded in otherwise ordinary error/log text."""

    def replace_assignment(match: re.Match[str]) -> str:
        raw_value = match.group("value")
        quote = (
            raw_value[0]
            if len(raw_value) >= 2
            and raw_value[0] in {'"', "'"}
            and raw_value[-1] == raw_value[0]
            else ""
        )
        replacement = f"{quote}<redacted-secret>{quote}"
        return match.group("prefix") + replacement

    def replace_basic(match: re.Match[str]) -> str:
        """Redact only syntactically valid Basic user:password credentials."""
        token = match.group("value")
        unpadded = token.rstrip("=")
        padded = unpadded + ("=" * (-len(unpadded) % 4))
        try:
            decoded = base64.b64decode(padded, validate=True)
        except (ValueError, TypeError):
            return match.group(0)
        if b":" not in decoded:
            return match.group(0)
        return match.group("prefix") + "<redacted-secret>"

    text = _SECRET_ASSIGNMENT.sub(replace_assignment, value)
    text = _URL_USERINFO.sub(r"\g<scheme><redacted-secret>@", text)
    text = _BASIC_VALUE.sub(replace_basic, text)
    text = _HF_TOKEN_VALUE.sub("<redacted-secret>", text)
    text = _BEARER_VALUE.sub("Bearer <redacted-secret>", text)
    text = _JWT_VALUE.sub("<redacted-secret>", text)
    return _PREFIXED_TOKEN_VALUE.sub("<redacted-secret>", text)


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
        if isinstance(o, str):
            return redact_secret_text(o)
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
