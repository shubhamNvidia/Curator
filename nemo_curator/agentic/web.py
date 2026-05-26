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
"""Tiny local web UI for the adaptive smart clarifier.

The flow is:

1. ``POST /api/plan`` — submit a prompt + dataset; the server runs the
   LLM intent extractor, profiles the source, then calls
   :func:`compose_smart_form` which returns:

   - ``inferred``: chips for everything the extractor already filled
     (collapsed into a single "Inferred …" panel at the top),
   - ``questions``: 1-5 LLM-phrased questions (with inline conditional
     follow-ups) for the *essential* fields that are still missing or
     uncertain,
   - ``advanced_form``: the full legacy ingredient-picker form, served
     so the UI can reveal it behind an "Show all options" expander.

2. The user answers the (short) question list and submits.

3. ``POST /api/build`` — the server applies the answer diff, compiles via
   :func:`plan_from_intent`, and returns the YAML.

There is no multi-turn loop — the smart form is a single dynamic round.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import secrets
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parseaddr
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from nemo_curator.agentic.clarifier import apply_answers
from nemo_curator.agentic.compiler import compile_ir_to_yaml, write_compiled_yaml
from nemo_curator.agentic.deterministic_planner import (
    DeterministicPlanningError,
    plan_from_intent,
)
from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.ir import ClusterProfile, SourceSpec
from nemo_curator.agentic.llm import LLMClient
from nemo_curator.agentic.planner_dag import extract_intent
from nemo_curator.agentic.profiler import profile_source
from nemo_curator.agentic.registry import build_registry
from nemo_curator.agentic.smart_clarifier import compose_smart_form

logger = logging.getLogger(__name__)

# Where per-user session logs are written. Override with
# CURATOR_ADV_WEBSITE_DIR. Default falls back to <repo>/../website if the
# env var is unset and the parent path looks like the ADV layout, else to
# /tmp/curator_adv_website so the server never crashes on import.
_DEFAULT_WEBSITE_DIR = (
    Path(__file__).resolve().parents[3] / "website"
    if (Path(__file__).resolve().parents[3] / "website").parent.name == "adv_phase2"
    else Path("/tmp/curator_adv_website")
)


def _website_dir() -> Path:
    """Resolve and create the directory used for per-session logs."""

    raw = os.environ.get("CURATOR_ADV_WEBSITE_DIR", "").strip()
    path = Path(raw).expanduser().resolve() if raw else _DEFAULT_WEBSITE_DIR
    (path / "sessions").mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class WebDefaults:
    dataset: str | None = "/lustre/fsw/portfolios/maxine/users/shbhawsar/ADV/adv_phase2/test_data/input"
    kind: str = "directory"
    out_root: str = "/tmp/curator_adv_web"
    model: str = "qwen/qwen3-next-80b-a3b-instruct"
    cluster_cpus: int = 8
    cluster_gpus: int = 0
    cluster_gpu_memory_gb: float = 24.0


@dataclass
class SessionState:
    prompt: str
    source_uri: str
    source_kind: str
    target_dir: str
    intent_payload: dict[str, Any]
    cluster: ClusterProfile
    profile_payload: dict[str, Any] | None = None
    answers: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """One end-user's authenticated session.

    Each login creates a *directory* on disk shared by the whole session:

        sessions/<sanitized_email>__<user_id8>/
            events.jsonl                # append-only audit log
            runs/
                <run_id>/
                    prompt.txt          # original natural-language ask
                    smart_form.json     # questions + inferred values from /api/plan
                    answers.json        # user's answers submitted to /api/build
                    intent.json         # final intent after answers applied
                    pipeline.yaml       # compiled pipeline YAML
                    response.json       # full /api/build response (stages, tuner, ...)
    """

    user_id: str
    email: str
    api_key: str
    created_at: datetime
    session_dir: Path
    runs: dict[str, SessionState] = field(default_factory=dict)
    _llm: LLMClient | None = None
    _llm_model: str | None = None

    @property
    def events_log(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def runs_dir(self) -> Path:
        return self.session_dir / "runs"

    def run_dir(self, run_id: str) -> Path:
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_llm(self, model: str | None = None) -> LLMClient:
        """Return (and lazily build) an LLMClient bound to this user's key.

        If *model* changes from the cached client's model the client is
        rebuilt so per-request UI overrides take effect immediately.
        """

        if self._llm is not None and (model or None) == self._llm_model:
            return self._llm
        self._llm = _make_llm(model or "", api_key=self.api_key)
        self._llm_model = model or None
        return self._llm

    @property
    def masked_key(self) -> str:
        k = self.api_key or ""
        if len(k) <= 8:
            return "***"
        return f"{k[:6]}…{k[-4:]}"


def _make_llm(model: str, *, api_key: str | None = None) -> LLMClient:
    """Return an ``LLMClient`` where both tiers point to *model*.

    If *model* is empty the env-var / ``LLMTier`` defaults are used as-is.
    All planning calls use ``tier="synth"``; critics also use the same
    model because both tiers now share one backend. When *api_key* is
    supplied it overrides any ``NVIDIA_API_KEY`` / ``OPENAI_API_KEY`` in
    the process environment — required for per-user sessions so two
    users can have different keys without leaking state.
    """

    from nemo_curator.agentic.llm import LLMTier, default_tiers  # noqa: PLC0415

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if model:
        _ = default_tiers()  # validate env var presence without keeping it
        kwargs["tiers"] = LLMTier(synth=model, planner=model)
    return LLMClient(**kwargs)


_EMAIL_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_email_for_filename(email: str) -> str:
    cleaned = _EMAIL_SANITIZE_RE.sub("_", email.strip().lower())
    return cleaned[:80] or "anonymous"


def _looks_like_email(email: str) -> bool:
    name, addr = parseaddr(email or "")
    if not addr or "@" not in addr:
        return False
    local, _, domain = addr.partition("@")
    return bool(local) and "." in domain


def _looks_like_nvidia_key(key: str) -> bool:
    k = (key or "").strip()
    return k.startswith("nvapi-") and len(k) >= 24


class AgenticWebApp:
    """Request handler state shared across HTTP requests."""

    def __init__(self, defaults: WebDefaults) -> None:
        self.defaults = defaults
        self.registry = build_registry(cross_check_runtime=False, eager=False)
        # Anonymous fallback client (env-based key). Only used by /api/plan
        # when a request comes in without a logged-in user — that path is
        # now blocked at the request handler, so this stays as a safety
        # net for non-LLM endpoints / future read-only routes.
        self.llm = _make_llm(defaults.model)
        # Per-plan run state, keyed by run_id. Lives inside its owning
        # UserSession.runs as well, but kept here so /api/build can look
        # up by run_id without the client having to echo user_id back.
        # (We still validate that the run belongs to the calling user.)
        self.sessions: dict[str, SessionState] = {}
        # Authenticated user sessions. Keyed by user_id (opaque token
        # returned by /api/login). Held in memory only — restarting the
        # server logs everyone out.
        self.user_sessions: dict[str, UserSession] = {}
        self._users_lock = Lock()
        self.website_dir = _website_dir()
        logger.info("web: per-user session logs -> %s", self.website_dir / "sessions")

    # ------------------------------------------------------------------
    # Per-user audit log helper.
    # ------------------------------------------------------------------

    def _user_log(self, user: UserSession, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "user_id": user.user_id,
            "email": user.email,
            **fields,
        }
        try:
            user.session_dir.mkdir(parents=True, exist_ok=True)
            with user.events_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # noqa: BLE001
            logger.warning("user_log: failed to write %s for %s: %s", event, user.email, exc)

    def _save_run_file(
        self,
        user: UserSession,
        run_id: str,
        name: str,
        data: Any,
    ) -> Path | None:
        """Persist a per-run artifact under ``sessions/<user>/runs/<run_id>/<name>``.

        - ``.json`` names get ``json.dumps`` (pretty, ``default=str``).
        - ``.yaml`` / ``.yml`` / ``.txt`` get the raw string written verbatim.
        Returns the path on success, ``None`` on failure (logged at warning).
        """

        try:
            run_dir = user.run_dir(run_id)
            path = run_dir / name
            if name.endswith(".json"):
                path.write_text(
                    json.dumps(data, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                path.write_text(str(data), encoding="utf-8")
            return path
        except OSError as exc:  # noqa: BLE001
            logger.warning(
                "user_log: failed to save %s for %s/%s: %s",
                name, user.email, run_id, exc,
            )
            return None

    # ------------------------------------------------------------------
    # /api/login — register a per-user session with their own NGC key.
    # ------------------------------------------------------------------

    def login(self, payload: dict[str, Any]) -> dict[str, Any]:
        email = str(payload.get("email") or "").strip()
        api_key = str(payload.get("api_key") or "").strip()
        if not _looks_like_email(email):
            return _error("Please enter a valid email address.")
        if not _looks_like_nvidia_key(api_key):
            return _error("API key looks malformed (expected 'nvapi-...').")

        user_id = secrets.token_hex(16)
        session_name = f"{_sanitize_email_for_filename(email)}__{user_id[:8]}"
        session_dir = self.website_dir / "sessions" / session_name
        (session_dir / "runs").mkdir(parents=True, exist_ok=True)
        user = UserSession(
            user_id=user_id,
            email=email,
            api_key=api_key,
            created_at=datetime.now(timezone.utc),
            session_dir=session_dir,
        )
        with self._users_lock:
            self.user_sessions[user_id] = user

        self._user_log(
            user,
            "login",
            api_key=user.masked_key,
            session_dir=str(session_dir),
        )
        logger.info(
            "web: login email=%s user_id=%s key=%s session_dir=%s",
            email, user_id, user.masked_key, session_dir,
        )
        return {
            "status": "ok",
            "user_id": user_id,
            "email": email,
            "session_dir": str(session_dir),
        }

    def _user_or_error(self, payload: dict[str, Any]) -> tuple[UserSession | None, dict[str, Any] | None]:
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            return None, _error("Please sign in first.", status="unauthorized")
        user = self.user_sessions.get(user_id)
        if user is None:
            return None, _error("Your session expired. Please sign in again.", status="unauthorized")
        return user, None

    def logout(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id") or "").strip()
        with self._users_lock:
            user = self.user_sessions.pop(user_id, None)
        if user is None:
            return {"status": "ok"}
        for run_id in list(user.runs):
            self.sessions.pop(run_id, None)
        self._user_log(user, "logout")
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # /api/plan — extract intent + build the form.
    # ------------------------------------------------------------------

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        user, err = self._user_or_error(payload)
        if err is not None:
            return err
        assert user is not None
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            return _error("Please enter an audio curation request.")
        source_uri = str(payload.get("dataset") or self.defaults.dataset or "").strip()
        if not source_uri:
            return _error("Please provide a dataset path or manifest.")
        source_kind = str(payload.get("kind") or self.defaults.kind or "directory")
        if source_kind not in {"directory", "manifest"}:
            return _error("Dataset kind must be 'directory' or 'manifest'.")

        run_id = secrets.token_hex(8)
        target_dir = str(payload.get("out") or "").strip()
        if not target_dir:
            target_dir = str(Path(self.defaults.out_root).expanduser().resolve() / run_id)

        # Per-user LLM client. Honors the in-UI model override but always
        # uses *this user's* API key, never the process-wide fallback.
        model_override = str(payload.get("model") or self.defaults.model or "").strip()
        llm = user.get_llm(model_override or None)

        try:
            cluster = _cluster_from_payload(payload, self.defaults)
        except ValueError as exc:
            return _error(str(exc))

        self._save_run_file(user, run_id, "prompt.txt", prompt)
        self._save_run_file(
            user,
            run_id,
            "request.json",
            {
                "prompt": prompt,
                "source_uri": source_uri,
                "source_kind": source_kind,
                "model": model_override or "<default>",
                "target_dir": target_dir,
                "cluster": cluster.model_dump(mode="json"),
            },
        )
        self._user_log(
            user,
            "plan_start",
            run_id=run_id,
            prompt=prompt,
            source_uri=source_uri,
            source_kind=source_kind,
            model=model_override or "<default>",
        )
        t0 = time.monotonic()
        try:
            rough_intent, profile = self._extract_intent_and_profile(
                prompt, source_uri, source_kind, llm,
            )
            smart = compose_smart_form(
                prompt, rough_intent, profile=profile, llm=llm, tier="synth",
            )
        except Exception as exc:  # noqa: BLE001
            self._user_log(
                user,
                "plan_error",
                run_id=run_id,
                duration_s=round(time.monotonic() - t0, 3),
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=4),
            )
            raise
        state = SessionState(
            prompt=prompt,
            source_uri=source_uri,
            source_kind=source_kind,
            target_dir=target_dir,
            intent_payload=smart.intent.model_dump(mode="json"),
            cluster=cluster,
            profile_payload=profile.model_dump(mode="json") if profile else None,
        )
        self.sessions[run_id] = state
        user.runs[run_id] = state

        smart_form_payload = smart.model_dump(mode="json")
        self._save_run_file(user, run_id, "smart_form.json", smart_form_payload)
        self._save_run_file(user, run_id, "intent_initial.json", state.intent_payload)
        if state.profile_payload is not None:
            self._save_run_file(user, run_id, "profile.json", state.profile_payload)

        self._user_log(
            user,
            "plan_done",
            run_id=run_id,
            duration_s=round(time.monotonic() - t0, 3),
            n_questions=len(smart.questions or []),
            n_inferred=len(smart.inferred or []),
            target_dir=target_dir,
        )

        return {
            "status": "form_ready",
            "session_id": run_id,
            "smart_form": smart.model_dump(mode="json"),
            "cluster": cluster.model_dump(mode="json"),
        }

    # ------------------------------------------------------------------
    # /api/build — apply answers + compile YAML.
    # ------------------------------------------------------------------

    def build(self, payload: dict[str, Any]) -> dict[str, Any]:
        user, err = self._user_or_error(payload)
        if err is not None:
            return err
        assert user is not None
        run_id = str(payload.get("session_id") or "")
        state = self.sessions.get(run_id)
        if state is None or run_id not in user.runs:
            return _error("Session expired. Please submit the prompt again.", status="expired")

        incoming = payload.get("answers") or {}
        if not isinstance(incoming, dict):
            return _error("Answers must be an object of {intent_path: value}.")
        state.answers.update({str(k): v for k, v in incoming.items()})

        # Allow the user to amend cluster details at build time too.
        # The execution mode is no longer a user-facing input; the tuner
        # picks it based on whether the pipeline's GPU demand fits.
        if any(k in payload for k in ("cluster_cpus", "cluster_gpus", "cluster_gpu_memory_gb")):
            try:
                state.cluster = _cluster_from_payload(payload, self.defaults, base=state.cluster)
            except ValueError as exc:
                return _error(str(exc))

        intent = IntentCategories(**state.intent_payload)
        intent = apply_answers(intent, state.answers)
        state.intent_payload = intent.model_dump(mode="json")

        self._save_run_file(user, run_id, "answers.json", state.answers)
        self._save_run_file(user, run_id, "intent.json", state.intent_payload)
        self._user_log(
            user,
            "build_start",
            run_id=run_id,
            n_answers=len(state.answers),
            answers=state.answers,
        )
        t0 = time.monotonic()
        try:
            response = self._compile(run_id, intent)
        except Exception as exc:  # noqa: BLE001
            self._user_log(
                user,
                "build_error",
                run_id=run_id,
                duration_s=round(time.monotonic() - t0, 3),
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=4),
            )
            raise
        # Persist the compiled artifacts under the user's session dir so they
        # can review/audit later without digging through /tmp.
        yaml_text = response.get("yaml")
        if isinstance(yaml_text, str) and yaml_text:
            self._save_run_file(user, run_id, "pipeline.yaml", yaml_text)
        self._save_run_file(user, run_id, "response.json", response)
        self._save_run_file(
            user,
            run_id,
            "stages.json",
            {
                "stages": response.get("stages") or [],
                "tuner": response.get("tuner") or [],
                "executor_config": response.get("executor_config"),
                "cluster": response.get("cluster"),
            },
        )

        self._user_log(
            user,
            "build_done",
            run_id=run_id,
            duration_s=round(time.monotonic() - t0, 3),
            status=response.get("status"),
            n_stages=len(response.get("stages") or []),
            error=response.get("error"),
            artifacts_dir=str(user.run_dir(run_id)),
        )
        return response

    def index_html(self) -> str:
        from nemo_curator.agentic.llm import default_tiers  # noqa: PLC0415
        defaults = {
            "dataset": self.defaults.dataset or "",
            "kind": self.defaults.kind,
            "out": "",
            "model": self.defaults.model or "",
            "cluster_cpus": self.defaults.cluster_cpus,
            "cluster_gpus": self.defaults.cluster_gpus,
            "cluster_gpu_memory_gb": self.defaults.cluster_gpu_memory_gb,
        }
        return _HTML.replace("__DEFAULTS_JSON__", json.dumps(defaults))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_intent_and_profile(
        self,
        prompt: str,
        source_uri: str,
        source_kind: str,
        llm: LLMClient,
    ) -> tuple[IntentCategories, Any]:
        source = SourceSpec(kind=source_kind, uri=source_uri)  # type: ignore[arg-type]
        try:
            profile = profile_source(source, sample_limit=64)
        except Exception as exc:  # noqa: BLE001
            logger.warning("web.plan: profile_source failed: %s", exc)
            profile = None
        intent = extract_intent(prompt, profile, llm, tier="synth")
        if not intent.raw_prompt:
            intent = intent.model_copy(update={"raw_prompt": prompt})
        return intent, profile

    def _compile(self, session_id: str, intent: IntentCategories) -> dict[str, Any]:
        state = self.sessions[session_id]
        out_dir = Path(state.target_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = plan_from_intent(
                intent,
                source_uri=state.source_uri,
                source_kind=state.source_kind,
                target_dir=str(out_dir),
                registry=self.registry,
                cluster=state.cluster,
            )
        except DeterministicPlanningError as exc:
            return _error(f"The compiler rejected this pipeline: {exc}", status="rejected")
        except Exception as exc:  # noqa: BLE001
            return _error(f"Planning failed: {exc}", status="failed")

        yaml_text = compile_ir_to_yaml(result.ir, self.registry)
        ir_path = out_dir / "ir.validated.json"
        result.ir.write(ir_path)
        yaml_path = write_compiled_yaml(result.ir, self.registry, out_dir / "compiled.yaml")
        findings_path = out_dir / "findings.json"
        findings = [
            {
                "severity": f.severity.value,
                "code": f.code,
                "detail": f.detail,
                "stage_index": f.stage_index,
                "stage_name": f.stage_name,
            }
            for f in result.report.findings
        ]
        findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
        dry_run_path = out_dir / "dry_run.json"
        dry_run_path.write_text(
            json.dumps(result.dry_run.to_payload(), indent=2, default=str),
            encoding="utf-8",
        )
        intent_path = out_dir / "intent.json"
        intent_path.write_text(result.intent.model_dump_json(indent=2), encoding="utf-8")
        prompt_path = out_dir / "prompt.txt"
        prompt_path.write_text(state.prompt + "\n", encoding="utf-8")
        answers_path = out_dir / "clarification_answers.json"
        answers_path.write_text(json.dumps(state.answers, indent=2), encoding="utf-8")

        tuner_payload = []
        for stage_ref in result.ir.stages:
            resources = stage_ref.resources.to_resources_kwargs() if stage_ref.resources else {}
            hints = stage_ref.backend_hints.model_dump(exclude_none=True) if stage_ref.backend_hints else {}
            tuner_payload.append({
                "stage": stage_ref.stage,
                "resources": resources,
                "batch_size": stage_ref.batch_size,
                "backend_hints": hints,
                "reasons": stage_ref.tuner_reasons,
            })
        executor_config = (
            result.ir.executor_config.model_dump(mode="json")
            if result.ir.executor_config is not None
            else None
        )

        return {
            "status": "ready",
            "session_id": session_id,
            "message": "Pipeline compiled and dry-run validation passed.",
            "yaml": yaml_text,
            "intent": result.intent.model_dump(mode="json"),
            "stages": [s.stage for s in result.ir.stages],
            "findings": findings,
            "dry_run_has_errors": result.dry_run.has_errors,
            "executor_config": executor_config,
            "cluster": state.cluster.model_dump(mode="json"),
            "tuner": tuner_payload,
            "paths": {
                "out_dir": str(out_dir),
                "compiled_yaml": str(yaml_path),
                "ir": str(ir_path),
                "intent": str(intent_path),
                "findings": str(findings_path),
                "dry_run": str(dry_run_path),
                "prompt": str(prompt_path),
                "answers": str(answers_path),
            },
        }


def _error(message: str, *, status: str = "error") -> dict[str, Any]:
    return {"status": status, "error": message}


def _cluster_from_payload(
    payload: dict[str, Any],
    defaults: "WebDefaults",
    *,
    base: ClusterProfile | None = None,
) -> ClusterProfile:
    """Read CPU / GPU / GPU-memory fields off the form payload.

    Falls back to ``base`` for fields the user didn't touch, or to the
    server-side defaults when no prior cluster exists.
    """

    def _coerce_int(value: Any, fallback: int, *, minimum: int = 0) -> int:
        if value is None or value == "":
            return fallback
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            msg = f"Cluster field expected an integer, got {value!r}."
            raise ValueError(msg) from None
        if n < minimum:
            msg = f"Cluster field must be >= {minimum}, got {n}."
            raise ValueError(msg)
        return n

    def _coerce_float(value: Any, fallback: float, *, minimum: float = 0.0) -> float:
        if value is None or value == "":
            return fallback
        try:
            n = float(value)
        except (TypeError, ValueError):
            msg = f"Cluster field expected a number, got {value!r}."
            raise ValueError(msg) from None
        if n < minimum:
            msg = f"Cluster field must be >= {minimum}, got {n}."
            raise ValueError(msg)
        return n

    base_cpus = base.cpus if base else defaults.cluster_cpus
    base_gpus = base.gpus if base else defaults.cluster_gpus
    base_mem = base.gpu_memory_gb if base else defaults.cluster_gpu_memory_gb
    return ClusterProfile(
        cpus=max(1, _coerce_int(payload.get("cluster_cpus"), base_cpus, minimum=1)),
        gpus=_coerce_int(payload.get("cluster_gpus"), base_gpus, minimum=0),
        gpu_memory_gb=_coerce_float(payload.get("cluster_gpu_memory_gb"), base_mem, minimum=0.0),
    )


class RequestHandler(BaseHTTPRequestHandler):
    app: AgenticWebApp

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_text(self.app.index_html(), content_type="text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/login":
                response = self.app.login(payload)
            elif path == "/api/logout":
                response = self.app.logout(payload)
            elif path == "/api/plan":
                response = self.app.plan(payload)
            elif path == "/api/build":
                response = self.app.build(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        except Exception as exc:  # noqa: BLE001
            response = _error(str(exc), status="failed")
        self._send_json(response)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            msg = "Request body must be a JSON object."
            raise ValueError(msg)
        return data

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, *, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    defaults: WebDefaults | None = None,
) -> None:
    app = AgenticWebApp(defaults or WebDefaults())

    class BoundHandler(RequestHandler):
        pass

    BoundHandler.app = app
    httpd = ThreadingHTTPServer((host, port), BoundHandler)
    print(f"curator-adv web UI: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the curator-adv ingredient-picker web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--dataset", default=None, help="Default dataset path shown in the UI.")
    parser.add_argument("--kind", default="directory", choices=["directory", "manifest"])
    parser.add_argument("--out-root", default="/tmp/curator_adv_web")
    parser.add_argument(
        "--model",
        default="",
        help=(
            "NIM model name for all LLM calls, e.g. "
            "'nvidia/llama-3.3-nemotron-super-49b-v1'. "
            "Overrides CURATOR_ADV_SYNTH_MODEL. "
            "Leave blank to use the env-var / built-in default."
        ),
    )
    parser.add_argument("--cluster-cpus", type=int, default=8, help="Default schedulable CPU count shown in the form.")
    parser.add_argument("--cluster-gpus", type=int, default=0, help="Default schedulable GPU count shown in the form.")
    parser.add_argument(
        "--cluster-gpu-memory-gb",
        type=float,
        default=24.0,
        help="Default per-GPU memory in GB shown in the form.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = WebDefaults()
    defaults = WebDefaults(
        dataset=args.dataset if args.dataset else base.dataset,
        kind=args.kind,
        out_root=args.out_root,
        model=args.model or base.model,
        cluster_cpus=args.cluster_cpus,
        cluster_gpus=args.cluster_gpus,
        cluster_gpu_memory_gb=args.cluster_gpu_memory_gb,
    )
    serve(host=args.host, port=args.port, defaults=defaults)
    return 0


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NeMo Curator ADV — Audio Pipeline Builder</title>
  <style>
    /* ── NVIDIA dark design system ───────────────────────────────────── */
    :root {
      color-scheme: dark;
      --bg:           #0a0c10;
      --bg2:          #0f1117;
      --panel:        #13161d;
      --panel2:       #1a1e28;
      --border:       #252a36;
      --border-hi:    #2e3548;
      --ink:          #e8ecf5;
      --ink2:         #a8b0c2;
      --muted:        #606880;
      --accent:       #76b900;   /* NVIDIA green */
      --accent-dim:   #5a8f00;
      --accent-glow:  rgba(118,185,0,.18);
      --blue:         #1f93ff;
      --blue-dim:     #1466b8;
      --ok:           #76b900;
      --ok-bg:        rgba(118,185,0,.12);
      --warn:         #f5a623;
      --warn-bg:      rgba(245,166,35,.12);
      --err:          #f05252;
      --err-bg:       rgba(240,82,82,.12);
      --code-bg:      #090b0f;
      --chip-prompt:  rgba(118,185,0,.15);
      --chip-profile: rgba(31,147,255,.15);
      --chip-default: rgba(255,255,255,.06);
      --radius:       8px;
      --radius-lg:    12px;
      --shadow:       0 2px 12px rgba(0,0,0,.5);
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      color: var(--ink);
      background: var(--bg);
      -webkit-font-smoothing: antialiased;
    }

    /* ── layout ─────────────────────────────────────────────────────── */
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
    }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 0;
      overflow-y: auto;
      height: 100vh;
      position: sticky;
      top: 0;
    }
    .aside-header {
      padding: 20px 20px 16px;
      border-bottom: 1px solid var(--border);
    }
    .logo-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 4px;
    }
    .logo-dot {
      width: 28px; height: 28px;
      background: var(--accent);
      border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px; font-weight: 800; color: #000; letter-spacing: -0.5px;
      flex-shrink: 0;
      box-shadow: 0 0 14px rgba(118,185,0,.4);
    }
    .aside-title { font-size: 15px; font-weight: 700; color: var(--ink); }
    .aside-sub   { font-size: 11px; color: var(--muted); }
    .aside-body  { padding: 16px 20px; display: flex; flex-direction: column; gap: 14px; flex: 1; }
    .aside-footer { padding: 14px 20px; border-top: 1px solid var(--border); display: flex; gap: 10px; }

    main {
      display: grid;
      grid-template-rows: 56px 1fr;
      min-width: 0;
      background: var(--bg2);
    }

    /* ── topbar ─────────────────────────────────────────────────────── */
    .topbar {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      gap: 12px;
    }
    .topbar-left { display: flex; align-items: center; gap: 10px; }
    .topbar h1 { font-size: 15px; font-weight: 600; color: var(--ink); }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 500;
      background: var(--panel2);
      border: 1px solid var(--border);
      color: var(--ink2);
      transition: all .2s;
    }
    .status-pill.busy  { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }
    .status-pill.done  { border-color: var(--ok);     color: var(--ok);     background: var(--ok-bg); }
    .status-pill.error { border-color: var(--err);    color: var(--err);    background: var(--err-bg); }
    .status-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: currentColor;
      flex-shrink: 0;
    }
    .busy .status-dot  { animation: pulse 1.2s ease-in-out infinite; }

    /* ── form controls ──────────────────────────────────────────────── */
    label {
      display: block;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, select, textarea {
      width: 100%;
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 9px 11px;
      font: inherit;
      font-size: 13px;
      color: var(--ink);
      outline: none;
      transition: border-color .15s, box-shadow .15s;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    input::placeholder, textarea::placeholder { color: var(--muted); }
    textarea { min-height: 108px; resize: vertical; }
    select { cursor: pointer; appearance: auto; }
    p.hint {
      font-size: 11px;
      color: var(--muted);
      line-height: 1.45;
      margin-top: 6px;
    }

    /* ── buttons ────────────────────────────────────────────────────── */
    button {
      border: none;
      border-radius: var(--radius);
      padding: 9px 16px;
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all .15s;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    button.primary {
      background: var(--accent);
      color: #000;
      box-shadow: 0 0 18px rgba(118,185,0,.25);
    }
    button.primary:hover { background: #8fd100; box-shadow: 0 0 24px rgba(118,185,0,.4); }
    button.primary:active { transform: scale(.97); }
    button.secondary {
      background: var(--panel2);
      color: var(--ink2);
      border: 1px solid var(--border);
    }
    button.secondary:hover { border-color: var(--border-hi); color: var(--ink); }
    button:disabled { opacity: .4; cursor: not-allowed; }

    /* option buttons */
    button.option {
      width: 100%;
      text-align: left;
      background: var(--bg2);
      border: 1px solid var(--border);
      color: var(--ink);
      margin-top: 8px;
      padding: 10px 14px;
      font-weight: 400;
      border-radius: var(--radius);
      flex-direction: column;
      align-items: flex-start;
      gap: 2px;
    }
    button.option:hover  { border-color: var(--accent); background: rgba(118,185,0,.06); }
    button.option.selected {
      border-color: var(--accent);
      background: rgba(118,185,0,.10);
      box-shadow: 0 0 0 1px var(--accent);
    }
    .option strong { font-size: 13px; font-weight: 600; display: block; }
    .option span   { font-size: 12px; color: var(--ink2); line-height: 1.35; display: block; }

    /* ── fieldset / cluster ─────────────────────────────────────────── */
    fieldset.cluster-box {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 14px 14px;
    }
    fieldset.cluster-box legend {
      font-size: 11px; font-weight: 600; letter-spacing: .04em;
      text-transform: uppercase;
      color: var(--muted); padding: 0 6px;
    }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

    /* ── form body (right panel) ────────────────────────────────────── */
    .form-body {
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 18px 20px;
      transition: border-color .2s;
    }
    .section:hover { border-color: var(--border-hi); }
    .section h2 { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
    .section .summary { font-size: 13px; color: var(--ink2); margin-bottom: 14px; }

    /* ── questions ──────────────────────────────────────────────────── */
    .question {
      border-top: 1px solid var(--border);
      padding-top: 14px;
      margin-top: 14px;
    }
    .question:first-of-type { border-top: none; padding-top: 0; margin-top: 0; }
    .question h3 { font-size: 14px; font-weight: 600; margin-bottom: 3px; }
    .question .why { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .freeform { margin-top: 8px; }

    /* ── chips ──────────────────────────────────────────────────────── */
    .chip {
      display: inline-flex; align-items: center;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 11px; font-weight: 500;
      margin: 3px 4px 3px 0;
      border: 1px solid transparent;
      cursor: default;
    }
    .chip.prompt  { background: var(--chip-prompt);  color: var(--accent); border-color: rgba(118,185,0,.3); }
    .chip.profile { background: var(--chip-profile); color: var(--blue);   border-color: rgba(31,147,255,.3); }
    .chip.user    { background: rgba(245,166,35,.15); color: var(--warn);  border-color: rgba(245,166,35,.3); }
    .chip.default { background: var(--chip-default); color: var(--ink2);   border-color: var(--border); }
    .chips-row { display: flex; flex-wrap: wrap; margin: 10px 0 4px; }

    /* ── badges (stage names) ───────────────────────────────────────── */
    .badge {
      display: inline-flex; align-items: center;
      border-radius: 999px;
      padding: 3px 10px;
      font-size: 11px; font-weight: 600;
      background: var(--ok-bg);
      color: var(--ok);
      border: 1px solid rgba(118,185,0,.3);
      margin: 4px 6px 4px 0;
    }

    /* ── code block ─────────────────────────────────────────────────── */
    pre {
      white-space: pre-wrap; overflow: auto; max-height: 480px;
      margin: 14px 0 0;
      padding: 14px 16px;
      border-radius: var(--radius);
      background: var(--code-bg);
      border: 1px solid var(--border);
      color: #c8d4f0;
      font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
      font-size: 12px; line-height: 1.6;
    }

    /* ── tuner table ────────────────────────────────────────────────── */
    table.tuner-table {
      width: 100%; border-collapse: collapse;
      margin-top: 14px; font-size: 12px;
    }
    table.tuner-table th, table.tuner-table td {
      border-bottom: 1px solid var(--border);
      padding: 8px 10px; vertical-align: top; text-align: left;
    }
    table.tuner-table th {
      background: var(--panel2); color: var(--muted);
      font-weight: 600; font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
    }
    table.tuner-table tr:hover td { background: rgba(255,255,255,.02); }
    table.tuner-table td.reasons { color: var(--muted); line-height: 1.4; }

    /* ── misc ───────────────────────────────────────────────────────── */
    .paths { margin-top: 12px; color: var(--muted); font-size: 12px; word-break: break-all; }
    .warning-box {
      background: var(--err-bg); border: 1px solid rgba(240,82,82,.35);
      border-radius: var(--radius); padding: 14px 16px;
      color: var(--err); font-size: 13px; line-height: 1.5;
    }
    .muted { color: var(--muted); font-size: 11px; margin-left: 6px; font-weight: 400; }

    details.section { padding: 14px 20px; }
    details.section > summary {
      cursor: pointer; font-size: 14px; font-weight: 600;
      list-style: none; padding: 4px 0; user-select: none;
      display: flex; align-items: center; gap: 8px;
    }
    details.section > summary::-webkit-details-marker { display: none; }
    .summary-arrow {
      width: 16px; height: 16px;
      color: var(--muted); transition: transform .2s;
      flex-shrink: 0;
    }
    details.section[open] .summary-arrow { transform: rotate(90deg); }
    details.infer-card { background: rgba(31,147,255,.05); border-color: rgba(31,147,255,.2); }

    .smart-q.followup {
      margin-left: 16px; padding-left: 14px;
      border-left: 2px solid rgba(118,185,0,.3);
      border-top: none;
    }
    .smart-q.followup h3 { font-size: 13px; }
    .followups { margin-top: 8px; }
    .subsection { border-top: 1px solid var(--border); padding: 14px 0 4px; margin-top: 14px; }
    .subsection:first-of-type { border-top: none; padding-top: 0; margin-top: 0; }
    .subsection h3 { font-size: 14px; font-weight: 600; margin-bottom: 2px; }
    .actions-row { display: flex; gap: 10px; flex-wrap: wrap; }

    /* ── loading / animation ────────────────────────────────────────── */
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: .35; }
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* ── login overlay ─────────────────────────────────────────────── */
    .login-overlay {
      position: fixed; inset: 0; z-index: 9999;
      background: radial-gradient(circle at 20% 10%, #1c2840 0%, #0a0f1c 70%);
      display: flex; align-items: center; justify-content: center;
      padding: 24px;
    }
    .login-overlay.hidden { display: none; }
    .login-card {
      width: min(440px, 100%);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 30px 28px 26px;
      box-shadow: 0 30px 80px rgba(0,0,0,.45);
      animation: slideIn .35s ease;
    }
    .login-card .login-brand {
      display: flex; align-items: center; gap: 10px; margin-bottom: 4px;
    }
    .login-card .login-brand .logo-dot {
      width: 28px; height: 28px; border-radius: 8px;
      background: var(--accent); color: #fff;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 14px;
    }
    .login-card h1 { font-size: 20px; margin: 8px 0 6px; }
    .login-card p.sub { color: var(--muted); margin: 0 0 18px; font-size: 13px; }
    .login-card form { display: flex; flex-direction: column; gap: 12px; }
    .login-card label {
      display: flex; flex-direction: column; gap: 6px;
      font-size: 12px; color: var(--muted); font-weight: 600;
      letter-spacing: .02em; text-transform: uppercase;
    }
    .login-card input {
      width: 100%; padding: 10px 12px;
      background: var(--input-bg, #f7f8fa);
      border: 1px solid var(--border);
      border-radius: 8px;
      font: inherit; color: var(--text);
    }
    .login-card input:focus {
      outline: none; border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(60,120,255,.18);
    }
    .login-card .login-btn {
      margin-top: 8px;
      width: 100%; justify-content: center;
    }
    .login-card .login-error {
      color: #c0392b; font-size: 12.5px; min-height: 16px; margin: 0;
    }
    .login-card .login-help {
      margin: 14px 0 0; padding-top: 14px;
      border-top: 1px solid var(--border);
      color: var(--muted); font-size: 12px; line-height: 1.5;
    }
    .login-card .login-help a { color: var(--accent); text-decoration: none; }
    .login-card .login-help a:hover { text-decoration: underline; }

    /* ── auth chip in topbar ──────────────────────────────────────── */
    .auth-chip {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 4px 10px 4px 8px;
      border: 1px solid var(--border); border-radius: 999px;
      background: var(--surface-2, #f5f6f8); color: var(--text);
      font-size: 12.5px; margin-right: 10px;
    }
    .auth-chip .auth-dot {
      width: 8px; height: 8px; border-radius: 50%; background: #2ecc71;
    }
    .auth-chip button {
      background: none; border: none; color: var(--accent);
      cursor: pointer; padding: 0 0 0 6px; font-weight: 600; font-size: 12.5px;
    }
    .auth-chip button:hover { text-decoration: underline; }

    /* ── pipeline mind-map (build result) ──────────────────────────── */
    @keyframes flowDash    { to { stroke-dashoffset: -120; } }
    @keyframes flowPulse   { 0%,100% { opacity: 1; } 50% { opacity: .45; } }
    @keyframes flowPop     { from { opacity: 0; transform: translateY(10px) scale(.96); } to { opacity: 1; transform: translateY(0) scale(1); } }
    @keyframes flowWave    { 0%,100% { transform: scaleY(0.55); } 50% { transform: scaleY(1); } }
    @keyframes flowGlow    { 0%,100% { box-shadow: 0 8px 26px rgba(0,0,0,.06); } 50% { box-shadow: 0 14px 34px rgba(80,120,255,.18); } }

    .pipe-summary {
      display: flex; align-items: center; justify-content: space-between;
      gap: 16px; flex-wrap: wrap;
      padding: 16px 18px; margin: 0 0 14px;
      background:
        radial-gradient(circle at 100% 0%, rgba(118,185,0,.18), transparent 55%),
        linear-gradient(135deg, var(--nv-black) 0%, var(--nv-charcoal) 100%);
      color: #fff; border-radius: 14px;
      box-shadow: 0 14px 30px rgba(0,0,0,.22);
      position: relative; overflow: hidden;
    }
    .pipe-summary::after {
      content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px;
      background: linear-gradient(90deg, var(--nv-green), var(--nv-green-dark));
    }
    .pipe-summary h2 { margin: 0; font-size: 16px; font-weight: 700; letter-spacing: .01em; display: flex; align-items: center; gap: 8px; }
    .pipe-summary h2 svg { color: var(--nv-green); }
    .pipe-summary .sub { margin-top: 2px; color: #c9d1c4; font-size: 12.5px; }
    .pipe-summary .pipe-pills { display: flex; flex-wrap: wrap; gap: 6px; }
    .pipe-pill {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 11px;
      background: rgba(255,255,255,.08);
      border: 1px solid rgba(255,255,255,.14);
      border-radius: 999px;
      font-size: 12px; color: #f1f5ec; font-weight: 600; letter-spacing: .02em;
    }
    .pipe-pill svg { width: 12px; height: 12px; }
    .pipe-pill.green { background: rgba(118,185,0,.22); border-color: rgba(118,185,0,.45); color: #d4ec88; }
    .pipe-pill.amber { background: rgba(251,191,36,.18); border-color: rgba(251,191,36,.35); color: #ffd98a; }
    .pipe-pill.red   { background: rgba(231,76,60,.18); border-color: rgba(231,76,60,.35); color: #ffb3a8; }

    /* canvas + grid + animated arrows */
    .flow-wrap {
      position: relative; overflow-x: auto; overflow-y: hidden;
      padding: 24px 12px 36px;
      background-image:
        radial-gradient(circle at 0% 0%, rgba(118,185,0,.08), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(78,122,0,.06), transparent 55%),
        linear-gradient(180deg, #fbfdf6 0%, #f3f7e8 100%),
        radial-gradient(circle, rgba(31,51,0,.06) 1px, transparent 1.5px);
      background-size: auto, auto, auto, 22px 22px;
      background-position: 0 0, 0 0, 0 0, 0 0;
      border: 1px solid var(--nv-green-border, #dde2ee);
      border-radius: 16px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.5);
    }
    .flow-svg {
      position: absolute; inset: 0;
      width: 100%; height: 100%; pointer-events: none; z-index: 1;
    }
    .flow-svg path.connector {
      fill: none;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-dasharray: 6 6;
      animation: flowDash 1.4s linear infinite;
      opacity: .85;
    }
    .flow-svg circle.endpoint {
      r: 3.5; fill: currentColor;
    }
    .flow-grid {
      position: relative; z-index: 2;
      display: inline-flex; gap: 64px;
      align-items: stretch; padding: 8px 26px 14px;
      min-width: 100%;
    }
    .flow-card {
      position: relative;
      width: 200px;
      background:
        linear-gradient(180deg, #ffffff 0%, #f8fcec 100%);
      border: 1px solid var(--nv-green-border, #BDDB7E);
      border-radius: 14px;
      padding: 18px 14px 14px;
      box-shadow: 0 10px 26px rgba(31, 51, 0, .08);
      color: var(--nv-charcoal);
      display: flex; flex-direction: column; gap: 10px;
      animation: flowPop .45s cubic-bezier(.2,.7,.2,1) both;
      transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .flow-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 18px 38px rgba(31, 51, 0, .18);
      border-color: var(--nv-green);
    }
    .flow-card::before {
      content: ""; position: absolute; left: 0; top: 14px; bottom: 14px;
      width: 5px; border-radius: 0 6px 6px 0;
      background: linear-gradient(180deg, var(--nv-green) 0%, var(--nv-green-dark) 100%);
      box-shadow: 0 0 14px rgba(118, 185, 0, .35);
    }
    .flow-card.source { animation-name: flowPop, flowGlow; animation-duration: .45s, 4s; animation-iteration-count: 1, infinite; animation-delay: 0s, .8s; }
    .flow-card.sink   { animation-name: flowPop, flowGlow; animation-duration: .45s, 4s; animation-iteration-count: 1, infinite; animation-delay: 0s, 1.4s; }
    .flow-step {
      position: absolute; top: -10px; left: -10px;
      min-width: 24px; height: 24px; padding: 0 7px;
      background: var(--nv-charcoal);
      color: #fff; font-size: 12px; font-weight: 800;
      border-radius: 999px;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 3px 8px rgba(0,0,0,.28), 0 0 0 2px rgba(118,185,0,.55);
    }
    .flow-card.source .flow-step,
    .flow-card.sink   .flow-step {
      background: var(--nv-green-deep);
      box-shadow: 0 3px 8px rgba(0,0,0,.32), 0 0 0 2px var(--nv-green);
    }
    .flow-icon-wrap {
      display: flex; align-items: center; gap: 10px;
    }
    .flow-icon-box {
      position: relative;
      width: 40px; height: 40px; border-radius: 11px;
      background: linear-gradient(135deg, var(--nv-green) 0%, var(--nv-green-dark) 100%);
      display: flex; align-items: center; justify-content: center;
      color: #fff;
      box-shadow: 0 6px 14px rgba(31, 51, 0, .22), inset 0 0 0 1px rgba(255,255,255,.18);
    }
    .flow-icon-box::after {
      content: ""; position: absolute; inset: 0; border-radius: inherit;
      background: radial-gradient(circle at 30% 25%, rgba(255,255,255,.45), transparent 60%);
      pointer-events: none;
    }
    .flow-icon-box svg { width: 22px; height: 22px; position: relative; z-index: 1; }
    .flow-cat-label {
      font-size: 10.5px; font-weight: 800; letter-spacing: .09em;
      text-transform: uppercase;
      color: var(--nv-green-deep);
    }
    .flow-card h4 {
      margin: 0; font-size: 14px; font-weight: 700;
      color: var(--nv-charcoal); line-height: 1.25;
    }
    .flow-card .flow-sub {
      font-size: 11.5px;
      color: var(--nv-slate);
      opacity: .85;
      word-break: break-word; line-height: 1.4;
    }
    .flow-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
    .flow-chip {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 3px 9px;
      background: #ffffff;
      border: 1px solid var(--nv-green-border, #BDDB7E);
      border-radius: 999px;
      font-size: 10.5px;
      color: var(--nv-charcoal);
      font-weight: 700;
    }
    .flow-chip svg { width: 11px; height: 11px; color: var(--nv-green-dark); }
    .flow-chip.gpu  { background: rgba(118,185,0,.14); color: var(--nv-green-deep); border-color: var(--nv-green); }
    .flow-chip.gpu svg { color: var(--nv-green-deep); }
    .flow-chip.cpu  { background: #f4f4f3; color: var(--nv-charcoal); border-color: #cbd0c4; }
    .flow-chip.cpu svg { color: var(--nv-charcoal); }
    .flow-chip.batch{ background: rgba(31,51,0,.06); color: var(--nv-green-deep); border-color: var(--nv-green-border); }
    .flow-card.source .flow-icon-wrap { align-items: flex-end; }
    .flow-card.source .flow-wave {
      display: flex; align-items: end; gap: 2px; height: 22px; margin-left: 2px;
      color: var(--nv-green);
    }
    .flow-card.source .flow-wave span {
      width: 3px;
      background: currentColor;
      border-radius: 2px;
      transform-origin: bottom; animation: flowWave 1.4s ease-in-out infinite;
      box-shadow: 0 0 4px rgba(118,185,0,.5);
    }
    .flow-card.source .flow-wave span:nth-child(1){ height: 60%; animation-delay: 0s; }
    .flow-card.source .flow-wave span:nth-child(2){ height: 90%; animation-delay: .1s; }
    .flow-card.source .flow-wave span:nth-child(3){ height: 40%; animation-delay: .2s; }
    .flow-card.source .flow-wave span:nth-child(4){ height: 75%; animation-delay: .3s; }
    .flow-card.source .flow-wave span:nth-child(5){ height: 55%; animation-delay: .4s; }

    /* category palettes
       - --cat-from / --cat-to : bright gradient stops (icon tiles, gradient bars)
       - --cat-fg              : WCAG-AA-safe text color on white surfaces
       The CSS `color` property is bound to --cat-fg so `currentColor` (used by
       the source-card waveform and the .flow-cat-label) stays readable. */
    .cat-reader  { --cat-from:#f8b94d; --cat-to:#d56a14; --cat-fg:#8a4308; color:var(--cat-fg); }
    .cat-asr     { --cat-from:#4b8cff; --cat-to:#1e57d6; --cat-fg:#163ea0; color:var(--cat-fg); }
    .cat-vad     { --cat-from:#9a82ff; --cat-to:#5b34c9; --cat-fg:#3d1f97; color:var(--cat-fg); }
    .cat-segment { --cat-from:#c08bff; --cat-to:#7f3dc9; --cat-fg:#592091; color:var(--cat-fg); }
    .cat-speaker { --cat-from:#ff7ab6; --cat-to:#c8307e; --cat-fg:#8e1c5b; color:var(--cat-fg); }
    .cat-quality { --cat-from:#5ed2a6; --cat-to:#178a5d; --cat-fg:#0b5d3f; color:var(--cat-fg); }
    .cat-filter  { --cat-from:#a4d447; --cat-to:#598614; --cat-fg:#3b5c08; color:var(--cat-fg); }
    .cat-resample{ --cat-from:#4fc8c1; --cat-to:#10817a; --cat-fg:#0a5b56; color:var(--cat-fg); }
    .cat-format  { --cat-from:#6c7dff; --cat-to:#3a47c4; --cat-fg:#212e94; color:var(--cat-fg); }
    .cat-writer  { --cat-from:#7c8aa6; --cat-to:#3b486b; --cat-fg:#222b48; color:var(--cat-fg); }
    .cat-generic { --cat-from:#a3a9b5; --cat-to:#5d6573; --cat-fg:#3d434f; color:var(--cat-fg); }

    /* side panels under the flow */
    .pipe-side {
      display: grid; gap: 14px;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      margin-top: 16px;
    }
    @media (max-width: 900px) { .pipe-side { grid-template-columns: 1fr; } }
    /* NVIDIA palette tokens, reused across the visualization */
    :root {
      --nv-green:        #76B900;
      --nv-green-dark:   #4E7A00;
      --nv-green-deep:   #1F3300;
      --nv-green-bg:     #F4FBE5;
      --nv-green-bg-2:   #E7F4C7;
      --nv-green-border: #BDDB7E;
      --nv-black:        #0B0F08;
      --nv-charcoal:     #1A1F19;
      --nv-slate:        #2A3128;
    }

    .pipe-panel {
      position: relative; overflow: hidden;
      background: linear-gradient(135deg, var(--nv-green-bg) 0%, var(--nv-green-bg-2) 100%);
      border: 1px solid var(--nv-green-border);
      border-radius: 14px;
      padding: 20px 18px 18px;
      box-shadow: 0 8px 22px rgba(31, 51, 0, .07);
    }
    .pipe-panel::before {
      content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
      background: linear-gradient(90deg, var(--nv-green) 0%, var(--nv-green-dark) 100%);
    }
    .pipe-panel h3 {
      margin: 0 0 14px;
      font-size: 13px; font-weight: 800; letter-spacing: .08em;
      color: var(--nv-green-deep); text-transform: uppercase;
      display: flex; align-items: center; gap: 8px;
    }
    .pipe-panel h3 svg { width: 14px; height: 14px; color: var(--nv-green-dark); }
    .pipe-panel h3::before {
      content: ""; width: 8px; height: 8px; border-radius: 50%;
      background: var(--nv-green);
      box-shadow: 0 0 0 3px rgba(118, 185, 0, .22);
    }

    .pipe-highlights { display: flex; flex-wrap: wrap; gap: 10px; }
    .pipe-hl {
      display: inline-flex; align-items: center; gap: 10px;
      padding: 8px 14px 8px 8px;
      background: #ffffff;
      border: 1.5px solid var(--nv-green-border);
      border-radius: 999px;
      font-size: 13px;
      color: var(--nv-slate);
      font-weight: 600;
      box-shadow: 0 2px 6px rgba(31, 51, 0, .06);
      transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    .pipe-hl:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(31, 51, 0, .12);
      border-color: var(--nv-green);
    }
    .pipe-hl .hl-icon {
      width: 26px; height: 26px; border-radius: 50%;
      background: linear-gradient(135deg, var(--nv-green) 0%, var(--nv-green-dark) 100%);
      color: #fff; display: inline-flex; align-items: center; justify-content: center;
      box-shadow: 0 2px 4px rgba(31, 51, 0, .22);
    }
    .pipe-hl .hl-icon svg { width: 13px; height: 13px; }
    .pipe-hl .hl-label {
      color: var(--nv-charcoal);
      font-weight: 600;
    }
    .pipe-hl .hl-val {
      color: #ffffff;
      background: var(--nv-charcoal);
      font-weight: 700;
      padding: 3px 9px;
      border-radius: 6px;
      letter-spacing: .01em;
      font-variant-numeric: tabular-nums;
    }
    .pipe-paths {
      margin-top: 14px;
      padding: 10px 12px;
      background: #f3f5fc; border: 1px dashed var(--border, #d5dbeb);
      border-radius: 10px; font-size: 12px; color: var(--muted, #5b6472);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-all;
    }
    .pipe-paths strong { color: var(--ink, #1b2438); }

    /* advanced disclosure (yaml, raw stages, findings) */
    .pipe-advanced {
      margin-top: 14px;
      background: var(--panel, #fff);
      border: 1px solid var(--border, #e0e4ee);
      border-radius: 14px;
      padding: 4px 0;
    }
    .pipe-advanced > summary {
      list-style: none; cursor: pointer; padding: 12px 16px;
      font-size: 13px; font-weight: 600; color: var(--ink, #1b2438);
      display: flex; align-items: center; gap: 8px;
    }
    .pipe-advanced > summary::-webkit-details-marker { display: none; }
    .pipe-advanced > summary .chev {
      transition: transform .2s ease; display: inline-flex;
    }
    .pipe-advanced[open] > summary .chev { transform: rotate(90deg); }
    .pipe-advanced > .adv-body { padding: 0 16px 14px; }
    .pipe-advanced .adv-tabs {
      display: flex; gap: 6px; margin-bottom: 8px;
    }
    .pipe-advanced .adv-tab {
      padding: 6px 12px; border: 1px solid var(--border, #e0e4ee);
      background: var(--panel, #fff); border-radius: 999px;
      font-size: 12px; font-weight: 600; color: var(--muted, #5b6472);
      cursor: pointer;
    }
    .pipe-advanced .adv-tab.active {
      background: linear-gradient(135deg, var(--nv-green) 0%, var(--nv-green-dark) 100%);
      border-color: transparent; color: #fff;
    }
    .pipe-advanced pre {
      background: #0d1320; color: #e2ecff;
      padding: 12px 14px; border-radius: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11.5px; line-height: 1.5; max-height: 360px; overflow: auto;
      white-space: pre; margin: 0;
    }
    @keyframes progressBar {
      0%   { width: 0%; }
      30%  { width: 45%; }
      60%  { width: 72%; }
      90%  { width: 90%; }
      100% { width: 100%; }
    }
    .loader-wrap {
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: 20px; padding: 48px 24px;
      animation: slideIn .3s ease;
    }
    .loader-ring {
      width: 48px; height: 48px;
      border: 3px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .8s linear infinite;
    }
    .loader-label {
      font-size: 14px; font-weight: 600; color: var(--ink);
      text-align: center;
    }
    .loader-sub { font-size: 12px; color: var(--muted); text-align: center; }
    .progress-track {
      width: 220px; height: 3px;
      background: var(--border); border-radius: 999px; overflow: hidden;
    }
    .progress-fill {
      height: 100%; background: var(--accent);
      border-radius: 999px;
      animation: progressBar 8s ease-out forwards;
    }
    .step-list {
      display: flex; flex-direction: column; gap: 10px; width: 100%; max-width: 320px;
    }
    .step-item {
      display: flex; align-items: center; gap: 10px;
      padding: 10px 14px;
      border-radius: var(--radius);
      background: var(--panel);
      border: 1px solid var(--border);
      font-size: 13px; font-weight: 500;
      transition: all .2s;
    }
    .step-item.active  { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }
    .step-item.done    { color: var(--ok); border-color: rgba(118,185,0,.3); background: var(--ok-bg); }
    .step-item.pending { color: var(--muted); }
    .step-icon { width: 18px; height: 18px; flex-shrink: 0; }

    /* ── welcome screen ─────────────────────────────────────────────── */
    .welcome {
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 400px; text-align: center; gap: 16px;
      animation: slideIn .4s ease;
    }
    .welcome-icon {
      width: 64px; height: 64px;
      background: var(--accent-glow);
      border: 1px solid rgba(118,185,0,.3);
      border-radius: 16px;
      display: flex; align-items: center; justify-content: center;
      font-size: 28px;
    }
    .welcome h2 { font-size: 20px; font-weight: 700; }
    .welcome p  { font-size: 14px; color: var(--ink2); max-width: 360px; line-height: 1.6; }

    /* ── exec info pill ─────────────────────────────────────────────── */
    .exec-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
    .exec-pill {
      display: flex; align-items: center; gap: 5px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 11px; font-weight: 600;
      background: var(--panel2);
      border: 1px solid var(--border);
      color: var(--ink2);
    }
    .exec-pill.green { color: var(--ok); border-color: rgba(118,185,0,.3); background: var(--ok-bg); }
    .exec-pill.blue  { color: var(--blue); border-color: rgba(31,147,255,.3); background: rgba(31,147,255,.1); }

    /* ── datalist styling (Chrome/Edge only, Firefox ignores) ─────── */
    input::-webkit-calendar-picker-indicator { filter: invert(.6); cursor: pointer; }

    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { height: auto; position: static; border-right: none; border-bottom: 1px solid var(--border); }
    }
  </style>
</head>
<body>
  <!-- ── login overlay (shown first; hidden after successful sign-in) ── -->
  <div class="login-overlay" id="loginOverlay">
    <div class="login-card">
      <div class="login-brand">
        <div class="logo-dot">N</div>
        <strong>NeMo Curator ADV</strong>
      </div>
      <h1>Sign in to start a session</h1>
      <p class="sub">Each user runs with their own NVIDIA build / NGC API key. We don't store it on disk — it lives in memory for this session only.</p>
      <form id="loginForm" autocomplete="off">
        <label>Email
          <input id="loginEmail" type="email" required placeholder="you@nvidia.com" />
        </label>
        <label>NVIDIA build API key
          <input id="loginKey" type="password" required placeholder="nvapi-..." spellcheck="false" autocapitalize="off" autocorrect="off" />
        </label>
        <button class="primary login-btn" id="loginBtn" type="submit">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14L21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>
          Continue
        </button>
        <p class="login-error" id="loginError"></p>
      </form>
      <p class="login-help">Get a free key at <a href="https://build.nvidia.com/" target="_blank" rel="noopener noreferrer">build.nvidia.com</a>. Your activity is logged per-session under <code>website/sessions/</code>.</p>
    </div>
  </div>

  <div class="shell">
    <!-- ── sidebar ─────────────────────────────────────────────────── -->
    <aside>
      <div class="aside-header">
        <div class="logo-row">
          <div class="logo-dot">N</div>
          <div>
            <div class="aside-title">NeMo Curator ADV</div>
            <div class="aside-sub">Audio Pipeline Builder</div>
          </div>
        </div>
      </div>

      <div class="aside-body">
        <div>
          <label for="dataset">Dataset path</label>
          <input id="dataset" placeholder="/path/to/audio_folder_or_manifest.json" />
        </div>
        <div>
          <label for="kind">Dataset kind</label>
          <select id="kind">
            <option value="directory">Directory</option>
            <option value="manifest">Manifest (JSON)</option>
          </select>
        </div>
        <div>
          <label for="out">Output directory</label>
          <input id="out" placeholder="/tmp/curator_adv_web/run" />
        </div>
        <div>
          <label for="model">LLM model</label>
          <select id="model">
            <option value="qwen/qwen3-next-80b-a3b-instruct">Qwen3 80B MoE — current default</option>
            <option value="moonshotai/kimi-k2.6">Kimi K2 — 1T MoE, best structured JSON on NIM</option>
            <option value="nvidia/llama-3.3-nemotron-super-49b-v1">Nemotron Super 49B — best quality/speed balance</option>
            <option value="nvidia/llama-3.3-nemotron-super-49b-v1.5">Nemotron Super 49B v1.5 — latest NVIDIA tuned</option>
            <option value="meta/llama-3.3-70b-instruct">Llama 3.3 70B — reliable, fast, great JSON</option>
            <option value="nvidia/llama-3.1-nemotron-70b-instruct">Nemotron 70B — NVIDIA-tuned, very fast</option>
            <option value="meta/llama-3.1-70b-instruct">Llama 3.1 70B — solid fallback</option>
            <option value="nvidia/llama-3.1-nemotron-51b-instruct">Nemotron 51B — fast mid-size</option>
            <option value="mistralai/mistral-large-2-instruct">Mistral Large 2 — strong instruction following</option>
            <option value="mistralai/mistral-large-3-675b-instruct-2512">Mistral Large 3 675B — maximum quality</option>
            <option value="meta/llama-3.1-8b-instruct">Llama 3.1 8B — dev / quick iteration only</option>
          </select>
          <p class="hint">Pick a NIM-hosted model. Click the dropdown to see all options.</p>
        </div>
        <fieldset class="cluster-box">
          <legend>Cluster resources</legend>
          <div class="grid-2">
            <div>
              <label for="cluster_cpus">CPUs</label>
              <input id="cluster_cpus" type="number" min="1" step="1" />
            </div>
            <div>
              <label for="cluster_gpus">GPUs</label>
              <input id="cluster_gpus" type="number" min="0" step="1" />
            </div>
            <div style="grid-column:1/-1">
              <label for="cluster_gpu_memory_gb">GPU memory (GB per GPU)</label>
              <input id="cluster_gpu_memory_gb" type="number" min="0" step="0.5" />
            </div>
          </div>
          <p class="hint">Streaming vs batch is decided automatically based on VRAM availability.</p>
        </fieldset>
        <div>
          <label for="prompt">Describe your goal</label>
          <textarea id="prompt" placeholder="Example: Clean my voice recordings, keep single-speaker clips under 60 s, and export a TTS dataset at 24 kHz mono."></textarea>
        </div>
      </div>

      <div class="aside-footer">
        <button class="primary" id="plan" style="flex:1">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          Plan pipeline
        </button>
        <button class="secondary" id="reset">Reset</button>
      </div>
    </aside>

    <!-- ── main area ───────────────────────────────────────────────── -->
    <main>
      <div class="topbar">
        <div class="topbar-left">
          <h1>Pipeline builder</h1>
        </div>
        <div style="display:flex; align-items:center; gap:10px">
          <span class="auth-chip" id="authChip" style="display:none">
            <span class="auth-dot"></span>
            <span id="authEmail"></span>
            <button id="logoutBtn" type="button">Sign out</button>
          </span>
          <div class="status-pill" id="statusPill">
            <span class="status-dot"></span>
            <span id="statusText">Ready</span>
          </div>
        </div>
      </div>
      <section class="form-body" id="formArea">
        <div class="welcome">
          <div class="welcome-icon">&#127911;</div>
          <h2>Build an audio pipeline</h2>
          <p>Describe your audio curation goal in the sidebar and press <strong>Plan pipeline</strong>. We'll profile your dataset, extract intent, and surface only the decisions you still need to make.</p>
        </div>
      </section>
    </main>
  </div>
  <script>
    const defaults = __DEFAULTS_JSON__;
    const formArea  = document.getElementById("formArea");
    const statusPill = document.getElementById("statusPill");
    const statusText = document.getElementById("statusText");

    /* ── model picker ────────────────────────────────────── */
    const NIM_MODELS = [
      { slug: "nvidia/llama-3.3-nemotron-super-49b-v1",  label: "Nemotron Super 49B",   badge: "balance", meta: "Best quality/speed" },
      { slug: "meta/llama-3.3-70b-instruct",             label: "Llama 3.3 70B",         badge: "balance", meta: "Great structured output" },
      { slug: "nvidia/llama-3.1-nemotron-70b-instruct",  label: "Nemotron 70B",          badge: "fast",    meta: "NVIDIA-tuned, very fast" },
      { slug: "meta/llama-3.1-70b-instruct",             label: "Llama 3.1 70B",         badge: "fast",    meta: "Reliable & fast" },
      { slug: "meta/llama-3.1-8b-instruct",              label: "Llama 3.1 8B",          badge: "dev",     meta: "Dev / iteration only" },
      { slug: "qwen/qwen3-next-80b-a3b-instruct",        label: "Qwen3 80B MoE",         badge: "dev",     meta: "Default (slow cold-start)" },
    ];

    let sessionId = null;
    let currentForm = null;
    let pendingAnswers = {};
    let pendingFreeform = {};
    let revealedFollowups = {};
    let pickedOptions = {};

    /* ── auth state (per-user session with their own NGC key) ────── */
    const AUTH_KEY = "curator_adv_user_v1";
    let userId   = null;
    let userEmail = null;
    function loadStoredAuth() {
      try {
        const raw = localStorage.getItem(AUTH_KEY);
        if (!raw) return null;
        return JSON.parse(raw);
      } catch (_) { return null; }
    }
    function saveAuth(uid, email) {
      try { localStorage.setItem(AUTH_KEY, JSON.stringify({user_id: uid, email})); } catch (_) {}
    }
    function clearAuth() {
      try { localStorage.removeItem(AUTH_KEY); } catch (_) {}
      userId = null; userEmail = null;
    }
    function showAuthChip() {
      const chip  = document.getElementById("authChip");
      const label = document.getElementById("authEmail");
      if (userEmail) {
        label.textContent = userEmail;
        chip.style.display = "inline-flex";
      } else {
        chip.style.display = "none";
      }
    }
    function showLogin(errMsg) {
      document.getElementById("loginOverlay").classList.remove("hidden");
      const err = document.getElementById("loginError");
      err.textContent = errMsg || "";
      setTimeout(() => document.getElementById("loginEmail").focus(), 30);
    }
    function hideLogin() {
      document.getElementById("loginOverlay").classList.add("hidden");
    }
    async function tryLogin(email, apiKey) {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({email, api_key: apiKey})
      });
      const data = await res.json();
      if (data.status === "ok") {
        userId = data.user_id; userEmail = data.email || email;
        saveAuth(userId, userEmail);
        showAuthChip(); hideLogin();
        setIdle();
        return true;
      }
      throw new Error(data.error || "Login failed");
    }
    async function doLogout() {
      const uid = userId;
      clearAuth();
      showAuthChip();
      resetAll();
      showLogin("");
      if (uid) {
        try {
          await fetch("/api/logout", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({user_id: uid})
          });
        } catch (_) {}
      }
    }

    /* ── status pill helpers ─────────────────────────────── */
    function setStatus(text, mode) {
      statusText.textContent = text;
      statusPill.className = "status-pill" + (mode ? " " + mode : "");
    }
    function setBusy(text) { setStatus(text, "busy"); }
    function setDone(text) { setStatus(text || "Done", "done"); }
    function setError(text){ setStatus(text || "Error", "error"); }
    function setIdle()     { setStatus("Ready"); }

    /* ── animated loading screen ─────────────────────────── */
    const PLAN_STEPS = [
      { id:"profile",  label:"Profiling dataset",         sub:"Sampling audio files…" },
      { id:"intent",   label:"Analysing prompt",          sub:"Extracting intent with LLM…" },
      { id:"clarify",  label:"Preparing smart form",      sub:"Selecting relevant questions…" },
      { id:"ready",    label:"Form ready",                sub:"" },
    ];
    const BUILD_STEPS = [
      { id:"compile",  label:"Compiling pipeline",        sub:"Applying your answers…" },
      { id:"tune",     label:"Tuning parameters",         sub:"Matching card defaults…" },
      { id:"validate", label:"Validating stages",         sub:"Checking shape compatibility…" },
      { id:"emit",     label:"Emitting YAML",             sub:"Writing compiled pipeline…" },
    ];

    function showLoader(steps, activeIdx) {
      const stepHtml = steps.map((s, i) => {
        const cls = i < activeIdx ? "done" : i === activeIdx ? "active" : "pending";
        const icon = i < activeIdx
          ? `<svg class="step-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`
          : i === activeIdx
          ? `<svg class="step-icon" style="animation:spin .7s linear infinite" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10" stroke-dasharray="32 10"/></svg>`
          : `<svg class="step-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="8"/></svg>`;
        return `<div class="step-item ${cls}">${icon}${escapeHtml(s.label)}</div>`;
      }).join("");
      formArea.innerHTML = `
        <div class="loader-wrap">
          <div class="loader-ring"></div>
          <div class="loader-label">${escapeHtml(steps[activeIdx]?.label || "Working…")}</div>
          <div class="loader-sub">${escapeHtml(steps[activeIdx]?.sub || "")}</div>
          <div class="progress-track"><div class="progress-fill"></div></div>
          <div class="step-list">${stepHtml}</div>
        </div>`;
    }

    function escapeHtml(text) {
      return String(text == null ? "" : text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }
    async function post(path, payload) {
      const body = Object.assign({}, payload || {});
      if (userId && !body.user_id) body.user_id = userId;
      const res = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (data && data.status === "unauthorized") {
        clearAuth(); showAuthChip(); showLogin("Your session expired. Please sign in again.");
        throw new Error(data.error || "unauthorized");
      }
      return data;
    }
    function settings() {
      return {
        dataset: document.getElementById("dataset").value.trim(),
        kind: document.getElementById("kind").value,
        out: document.getElementById("out").value.trim(),
        model: document.getElementById("model").value.trim(),
        prompt: document.getElementById("prompt").value.trim(),
        cluster_cpus: document.getElementById("cluster_cpus").value.trim(),
        cluster_gpus: document.getElementById("cluster_gpus").value.trim(),
        cluster_gpu_memory_gb: document.getElementById("cluster_gpu_memory_gb").value.trim()
      };
    }
    async function planFlow() {
      const s = settings();
      if (!s.prompt) { setBusy("Enter a prompt first"); return; }
      setBusy("Analysing prompt…");
      showLoader(PLAN_STEPS, 0);
      // animate through steps while waiting
      let step = 0;
      const stepTimer = setInterval(() => {
        step = Math.min(step + 1, PLAN_STEPS.length - 2);
        showLoader(PLAN_STEPS, step);
        setBusy(PLAN_STEPS[step].label + "…");
      }, 2200);
      try {
        const data = await post("/api/plan", s);
        clearInterval(stepTimer);
        showLoader(PLAN_STEPS, PLAN_STEPS.length - 1);
        await new Promise(r => setTimeout(r, 350));
        handlePlanResponse(data);
      } catch (err) {
        clearInterval(stepTimer);
        setError("Network error");
        formArea.innerHTML = `<div class="section warning-box">${escapeHtml(err.message)}</div>`;
      }
    }
    function handlePlanResponse(data) {
      if (data.status !== "form_ready") {
        setError("Planning failed");
        formArea.innerHTML = `<div class="section warning-box">${escapeHtml(data.error || "Unknown error")}</div>`;
        return;
      }
      sessionId = data.session_id;
      currentForm = data.smart_form;
      pendingAnswers = {};
      pendingFreeform = {};
      revealedFollowups = {};
      pickedOptions = {};
      const hasQ = currentForm.questions && currentForm.questions.length;
      setDone(hasQ ? "Answer the questions below" : "Ready to build");
      renderSmartForm(currentForm);
    }
    function renderSmartForm(form) {
      let html = "";

      // 1. Inferred-from-prompt chips, collapsed by default.
      const inferred = form.inferred || [];
      if (inferred.length) {
        const chipsHtml = inferred.map(c => {
          const src = (c.source || "default").toLowerCase();
          return `<span class="chip ${escapeHtml(src)}" title="${escapeHtml(c.why || '')}">${escapeHtml(c.label)}</span>`;
        }).join("");
        html += `
          <details class="section infer-card" style="animation:slideIn .3s ease">
            <summary>
              <svg class="summary-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
              <strong>Inferred ${inferred.length} setting${inferred.length === 1 ? '' : 's'} from your prompt + dataset</strong>
              <span class="muted">(click to review)</span>
            </summary>
            <div class="chips-row">${chipsHtml}</div>
            <p class="hint">These are pre-filled. Override via the questions below or "Show all options".</p>
          </details>`;
      }

      // 2. The dynamic questions.
      const questions = form.questions || [];
      if (questions.length === 0) {
        html += `<div class="section" style="animation:slideIn .3s ease"><h2>&#10003; Nothing left to ask</h2><p class="summary">We have everything we need. Press <strong>Build pipeline</strong> to compile.</p></div>`;
      } else {
        html += `<div class="section" style="animation:slideIn .3s ease"><h2>Quick questions</h2><p class="summary">Answer the essentials below — we'll fill in the rest from sensible defaults.</p>`;
        questions.forEach(q => {
          html += renderSmartQuestion(q, /*depth=*/0);
        });
        html += `</div>`;
      }

      // 3. Advanced expander
      html += `
        <details class="section">
          <summary>
            <svg class="summary-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
            <strong>Show all options</strong>
            <span class="muted">— every ingredient, no filtering</span>
          </summary>
          <div id="advancedMount"></div>
        </details>`;

      // 4. Actions
      html += `
        <div class="actions-row">
          <button class="primary" id="build">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Build pipeline
          </button>
          <button class="secondary" id="reset2">Reset</button>
        </div>`;

      formArea.innerHTML = html;
      document.getElementById("build").addEventListener("click", buildFlow);
      document.getElementById("reset2").addEventListener("click", resetAll);
      wireSmartFormInteractions();
      mountAdvancedFormOnExpand(form.advanced_form);
    }
    function renderSmartQuestion(q, depth) {
      const isFollowup = depth > 0;
      const hidden = q.__hidden ? 'style="display:none"' : '';
      let html = `<div class="question smart-q ${isFollowup ? 'followup' : ''}" data-qid="${escapeHtml(q.id)}" data-intent-path="${escapeHtml(q.intent_path)}" ${hidden}>
        <h3>${escapeHtml(q.title)}</h3>`;
      if (q.detail) {
        html += `<p class="why">${escapeHtml(q.detail)}</p>`;
      }
      (q.options || []).forEach(o => {
        const sel = pickedOptions[q.id] === o.id ? 'selected' : '';
        html += `<button class="option ${sel}" data-qid="${escapeHtml(q.id)}" data-oid="${escapeHtml(o.id)}">
          <strong>${escapeHtml(o.label)}</strong>
          ${o.description ? `<span>${escapeHtml(o.description)}</span>` : ""}
        </button>`;
        if (o.is_freeform) {
          html += `<div class="freeform"><input type="text" data-freeform-qid="${escapeHtml(q.id)}" placeholder="${escapeHtml(o.freeform_placeholder || '')}"></div>`;
        }
      });
      // Render follow-ups in their own container; visibility toggled by parent option click.
      if (q.follow_ups && q.follow_ups.length) {
        html += `<div class="followups" data-parent="${escapeHtml(q.id)}">`;
        q.follow_ups.forEach(fu => {
          const fuMarked = Object.assign({}, fu, { __hidden: !isFollowupVisible(q.id, fu.id) });
          html += renderSmartQuestion(fuMarked, depth + 1);
        });
        html += `</div>`;
      }
      return html + `</div>`;
    }
    function isFollowupVisible(parentQid, followupQid) {
      const revealedHere = revealedFollowups[parentQid] || new Set();
      return revealedHere.has(followupQid);
    }
    function wireSmartFormInteractions() {
      formArea.querySelectorAll("button.option").forEach(btn => {
        btn.addEventListener("click", () => {
          handleSmartOptionClick(btn.dataset.qid, btn.dataset.oid, btn);
        });
      });
      formArea.querySelectorAll("input[data-freeform-qid]").forEach(input => {
        input.addEventListener("input", () => {
          const qid = input.dataset.freeformQid;
          pendingFreeform[qid] = input.value;
          // Re-apply the currently selected freeform option's value, if any.
          const selectedBtn = formArea.querySelector(`button.option.selected[data-qid="${qid}"]`);
          if (selectedBtn) {
            const q = findSmartQuestion(qid);
            const o = q && (q.options || []).find(x => x.id === selectedBtn.dataset.oid);
            if (o && o.is_freeform) {
              const v = parseFreeform(input.value, o.freeform_kind);
              if (v != null) recordAnswer(q, o, v);
            }
          }
        });
      });
    }
    function handleSmartOptionClick(qid, oid, btn) {
      const q = findSmartQuestion(qid);
      if (!q) return;
      const o = (q.options || []).find(x => x.id === oid);
      if (!o) return;

      // Visual selection state, only inside this question's own scope.
      const scope = btn.closest(".smart-q");
      scope.querySelectorAll(`:scope > button.option`).forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      pickedOptions[qid] = oid;

      let value = o.value;
      if (o.is_freeform) {
        value = parseFreeform(pendingFreeform[qid], o.freeform_kind);
        if (value == null) {
          setBusy("Enter a value in the custom field above");
          return;
        }
      }
      recordAnswer(q, o, value);
      updateFollowupVisibility(q, o, scope);
    }
    function recordAnswer(q, o, value) {
      // Skip writing meta-paths (those starting with __) to the answers map.
      if (q.intent_path && !q.intent_path.startsWith("__") && value !== null && value !== undefined) {
        pendingAnswers[q.intent_path] = value;
      }
      if (o.apply && typeof o.apply === "object") {
        for (const path in o.apply) {
          pendingAnswers[path] = o.apply[path];
        }
      }
      setDone("Captured: " + (q.title || q.intent_path));
    }
    function updateFollowupVisibility(q, o, scope) {
      const followupsContainer = scope.querySelector(`:scope > .followups`);
      if (!followupsContainer) return;
      const reveals = new Set(o.reveals || []);
      revealedFollowups[q.id] = reveals;
      // Toggle each direct child question.
      followupsContainer.querySelectorAll(`:scope > .smart-q`).forEach(child => {
        const childId = child.dataset.qid;
        const visible = reveals.has(childId);
        child.style.display = visible ? '' : 'none';
        if (!visible) {
          // Clear answers for hidden follow-ups so we don't submit stale data.
          const ip = child.dataset.intentPath;
          if (ip && !ip.startsWith("__")) delete pendingAnswers[ip];
        }
      });
    }
    function findSmartQuestion(qid) {
      function walk(qs) {
        for (const q of (qs || [])) {
          if (q.id === qid) return q;
          const found = walk(q.follow_ups);
          if (found) return found;
        }
        return null;
      }
      return walk(currentForm && currentForm.questions);
    }
    function mountAdvancedFormOnExpand(advancedForm) {
      const details = formArea.querySelector("details.section:last-of-type");
      if (!details || !advancedForm) return;
      const mount = details.querySelector("#advancedMount");
      let mounted = false;
      details.addEventListener("toggle", () => {
        if (details.open && !mounted) {
          mount.innerHTML = renderAdvancedForm(advancedForm);
          wireAdvancedFormInteractions(mount, advancedForm);
          mounted = true;
        }
      });
    }
    function renderAdvancedForm(advancedForm) {
      let html = '<p class="hint" style="margin-bottom:12px">Every available ingredient — overrides smart-form answers.</p>';
      (advancedForm.sections || []).forEach(section => {
        html += `<div class="subsection"><h3>${escapeHtml(section.title)}</h3><p class="summary">${escapeHtml(section.summary)}</p>`;
        (section.questions || []).forEach(q => {
          if (!q.visible) return;
          html += renderAdvancedQuestion(q);
        });
        html += `</div>`;
      });
      return html;
    }
    function renderAdvancedQuestion(q) {
      const prefill = q.prefill;
      const chip = prefill
        ? `<span class="chip ${escapeHtml(prefill.source)}">${escapeHtml(prefill.source)}: ${escapeHtml(JSON.stringify(prefill.value))}</span>`
        : `<span class="chip default">unset</span>`;
      let html = `<div class="question adv-q" data-qid="${escapeHtml(q.id)}">
        <h4>${escapeHtml(q.title)} ${chip}</h4>
        <p class="why">${escapeHtml(q.why)}</p>`;
      (q.options || []).forEach(o => {
        html += `<button class="option" data-adv-qid="${escapeHtml(q.id)}" data-adv-oid="${escapeHtml(o.id)}" data-adv-path="${escapeHtml(q.intent_path)}">
          <strong>${escapeHtml(o.label)}</strong>
          ${o.description ? `<span>${escapeHtml(o.description)}</span>` : ""}
        </button>`;
        if (o.is_freeform) {
          html += `<div class="freeform"><input type="text" data-adv-freeform-qid="${escapeHtml(q.id)}" placeholder="${escapeHtml(o.freeform_placeholder || '')}"></div>`;
        }
      });
      return html + `</div>`;
    }
    function wireAdvancedFormInteractions(mount, advancedForm) {
      const findAdvQ = (qid) => {
        for (const section of (advancedForm.sections || [])) {
          for (const q of section.questions) { if (q.id === qid) return q; }
        }
        return null;
      };
      mount.querySelectorAll("input[data-adv-freeform-qid]").forEach(input => {
        input.addEventListener("input", () => {
          pendingFreeform["adv:" + input.dataset.advFreeformQid] = input.value;
        });
      });
      mount.querySelectorAll("button.option").forEach(btn => {
        btn.addEventListener("click", () => {
          const qid = btn.dataset.advQid;
          const oid = btn.dataset.advOid;
          const path = btn.dataset.advPath;
          const q = findAdvQ(qid);
          if (!q) return;
          const o = (q.options || []).find(x => x.id === oid);
          if (!o) return;
          mount.querySelectorAll(`button.option[data-adv-qid="${qid}"]`).forEach(b => b.classList.remove("selected"));
          btn.classList.add("selected");
          let v = o.value;
          if (o.is_freeform) {
            v = parseFreeform(pendingFreeform["adv:" + qid], o.freeform_kind);
            if (v == null) { setBusy("Fill the custom field above first"); return; }
          }
          if (v !== null && v !== undefined) pendingAnswers[path] = v;
          if (o.apply && typeof o.apply === "object") {
            for (const p in o.apply) pendingAnswers[p] = o.apply[p];
          }
          setDone("Override: " + path + " = " + JSON.stringify(v));
        });
      });
    }
    function parseFreeform(raw, kind) {
      if (raw == null) return null;
      raw = String(raw).trim();
      if (!raw) return null;
      if (kind === "int" || kind === "int_hz" || kind === "speaker_count") {
        const n = parseInt(raw, 10);
        return Number.isFinite(n) ? n : null;
      }
      if (kind === "float" || kind === "float_sec") {
        const n = parseFloat(raw);
        return Number.isFinite(n) ? n : null;
      }
      if (kind === "duration_range") {
        const parts = raw.split(",").map(p => parseFloat(p));
        if (parts.length !== 2 || parts.some(p => !Number.isFinite(p))) return null;
        return { __apply__: { "segmentation.duration_min_sec": parts[0], "segmentation.duration_max_sec": parts[1] } };
      }
      return raw;
    }
    async function buildFlow() {
      setBusy("Compiling pipeline…");
      showLoader(BUILD_STEPS, 0);
      let step = 0;
      const stepTimer = setInterval(() => {
        step = Math.min(step + 1, BUILD_STEPS.length - 2);
        showLoader(BUILD_STEPS, step);
        setBusy(BUILD_STEPS[step].label + "…");
      }, 1600);
      try {
        const expandedAnswers = expandAnswers(pendingAnswers);
        const s = settings();
        const data = await post("/api/build", {
          session_id: sessionId,
          answers: expandedAnswers,
          cluster_cpus: s.cluster_cpus,
          cluster_gpus: s.cluster_gpus,
          cluster_gpu_memory_gb: s.cluster_gpu_memory_gb
        });
        clearInterval(stepTimer);
        showLoader(BUILD_STEPS, BUILD_STEPS.length - 1);
        await new Promise(r => setTimeout(r, 300));
        handleBuildResponse(data);
      } catch (err) {
        clearInterval(stepTimer);
        setError("Build failed");
        formArea.innerHTML = `<div class="section warning-box">${escapeHtml(err.message)}</div>`;
      }
    }
    function expandAnswers(answers) {
      const flat = {};
      for (const path in answers) {
        const v = answers[path];
        if (v && typeof v === "object" && v.__apply__) {
          for (const innerPath in v.__apply__) {
            flat[innerPath] = v.__apply__[innerPath];
          }
        } else {
          flat[path] = v;
        }
      }
      return flat;
    }
    /* ── pipeline mind-map ─────────────────────────────────────── */
    const ICONS = {
      input:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h4l2-3h6l2 3h4v12H3z"/><circle cx="12" cy="13" r="3.5"/></svg>',
      mic:      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v4M8 22h8"/></svg>',
      scissors: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.12 15.88M14.47 14.48 20 20M8.12 8.12 12 12"/></svg>',
      layers:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 2 7l10 5 10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>',
      users:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
      gauge:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 14l4-4"/><path d="M3.34 17A10 10 0 1 1 20.66 17"/><circle cx="12" cy="14" r="1.6" fill="currentColor"/></svg>',
      filter:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 3H2l8 9.46V19l4 2v-8.54z"/></svg>',
      refresh:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
      sliders:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
      save:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>',
      cog:      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9 1.65 1.65 0 0 0 4.27 7.18l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6 1.65 1.65 0 0 0 10 3.09V3a2 2 0 0 1 4 0v.09A1.65 1.65 0 0 0 15 4.6a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.36.51.92.85 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
      target:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
      cpu:      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>',
      bolt:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
      check:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
      route:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="19" r="3"/><circle cx="18" cy="5" r="3"/><path d="M9 19h6a3 3 0 0 0 0-6H9a3 3 0 0 1 0-6h6"/></svg>',
    };
    const STAGE_RULES = [
      { id:"reader",  cat:"cat-reader",   match:/(ManifestReader|JsonReader|JsonlReader|Source|DirectoryRead)/i, icon:ICONS.input,    label:"Read source" },
      { id:"writer",  cat:"cat-writer",   match:/(Writer|Sink|Save|Export|WriteOut)/i,                            icon:ICONS.save,     label:"Write output" },
      { id:"asr",     cat:"cat-asr",      match:/(Asr|Whisper|Parakeet|Transcrib|Stt|Caption|Wav2Vec)/i,         icon:ICONS.mic,      label:"Transcribe" },
      { id:"vad",     cat:"cat-vad",      match:/(Vad|Silero|SpeechDetect|VoiceActivity)/i,                       icon:ICONS.scissors, label:"Detect speech" },
      { id:"speaker", cat:"cat-speaker",  match:/(Speaker|Diariz|ResNet|ECAPA|TitaNet)/i,                         icon:ICONS.users,    label:"Speakers" },
      { id:"segment", cat:"cat-segment",  match:/(Segment|Chunk|Window|Alm|Pack|Slice)/i,                         icon:ICONS.layers,   label:"Segment" },
      { id:"quality", cat:"cat-quality",  match:/(Mos|Sigmos|Utmos|Band|Quality|Loudness|Snr|Pesq)/i,             icon:ICONS.gauge,    label:"Quality score" },
      { id:"filter",  cat:"cat-filter",   match:/(Preserve|Filter|Drop|Reject|Gate)/i,                            icon:ICONS.filter,   label:"Filter" },
      { id:"resample",cat:"cat-resample", match:/(Resample|Resamp|Rate|Sample)/i,                                 icon:ICONS.refresh,  label:"Resample" },
      { id:"format",  cat:"cat-format",   match:/(ChannelConvert|Channel|Encode|Format|Convert|Normalize|Trim|Pad)/i, icon:ICONS.sliders, label:"Format" },
    ];
    const PRETTY = {
      "ManifestReader":            "Read manifest",
      "JsonlReaderStage":          "Read JSONL manifest",
      "InferenceAsrNemoStage":     "Transcribe (NeMo)",
      "InferenceAsrWhisperStage":  "Transcribe (Whisper)",
      "ManifestWriterStage":       "Write manifest",
      "AlmDataBuilderStage":       "Build ALM windows",
      "SileroVadStage":            "Voice activity (Silero)",
      "ResampleStage":             "Resample",
      "ChannelConvertStage":       "Convert channels",
      "UtmosStage":                "UTMOS score",
      "SigmosStage":               "SIGMOS score",
      "BandPredictionStage":       "Band prediction",
      "SpeakerSeparationStage":    "Speaker separation",
      "DiarizationStage":          "Diarization",
      "PreserveByValueStage":      "Quality filter",
    };
    function categorizeStage(name) {
      for (const rule of STAGE_RULES) {
        if (rule.match.test(name)) return rule;
      }
      return { id:"generic", cat:"cat-generic", icon:ICONS.cog, label:"Process" };
    }
    function prettyStageName(name) {
      if (PRETTY[name]) return PRETTY[name];
      let s = String(name || "");
      s = s.replace(/Stage$/, "").replace(/^Inference/, "");
      s = s.replace(/([a-z])([A-Z])/g, "$1 $2");
      s = s.replace(/_/g, " ");
      return s.trim() || "Process";
    }
    function resourceChips(t) {
      const r = t.resources || {};
      const out = [];
      if (r.gpus != null && r.gpus > 0) {
        const g = Number(r.gpus);
        out.push(`<span class="flow-chip gpu">${ICONS.bolt} ${g % 1 === 0 ? g : g.toFixed(2)}× GPU</span>`);
      }
      if (r.cpus != null) {
        const c = Number(r.cpus);
        out.push(`<span class="flow-chip cpu">${ICONS.cpu} ${c % 1 === 0 ? c : c.toFixed(1)} CPU</span>`);
      }
      if (t.batch_size != null) {
        out.push(`<span class="flow-chip batch">batch ${t.batch_size}</span>`);
      }
      return out.join("");
    }
    function flowCardHtml(item, idx) {
      const cat = categorizeStage(item.stage);
      const pretty = prettyStageName(item.stage);
      const sub = item.sub || cat.label;
      const isSource = idx === 0;
      const isSink = item.__sink === true;
      const cls = ["flow-card", cat.cat, isSource ? "source" : "", isSink ? "sink" : ""].filter(Boolean).join(" ");
      const wave = isSource
        ? `<span class="flow-wave"><span></span><span></span><span></span><span></span><span></span></span>`
        : "";
      const chips = item.__chipsHtml || "";
      return `
        <div class="${cls}" style="animation-delay:${(idx*70)}ms" data-step="${idx+1}">
          <span class="flow-step">${idx+1}</span>
          <div class="flow-icon-wrap">
            <div class="flow-icon-box">${cat.icon}</div>
            ${wave}
            <div style="display:flex;flex-direction:column;gap:2px;">
              <span class="flow-cat-label">${escapeHtml(cat.label)}</span>
              <h4>${escapeHtml(pretty)}</h4>
            </div>
          </div>
          <div class="flow-sub" title="${escapeHtml(item.stage || "")}">${escapeHtml(sub)}</div>
          ${chips ? `<div class="flow-chips">${chips}</div>` : ""}
        </div>`;
    }
    function buildFlowItems(data) {
      const stages = data.stages || [];
      const tuner  = data.tuner || [];
      const tunerByName = {};
      for (const t of tuner) tunerByName[t.stage] = t;
      return stages.map((stageName) => {
        const t = tunerByName[stageName] || {};
        return {
          stage: stageName,
          sub: prettyStageName(stageName),
          __chipsHtml: resourceChips(t),
          __tuner: t,
        };
      });
    }
    function intentHighlights(intent) {
      const out = [];
      if (!intent) return out;
      const o = intent.output || {};
      const seg = intent.segmentation || {};
      const q = intent.quality || {};
      const sp = intent.speakers || {};
      const tx = intent.text || {};
      function add(catCls, icon, label, val) {
        out.push({catCls, icon, label, val});
      }
      if (o.sample_rate) add("cat-resample", ICONS.refresh, "Sample rate", `${o.sample_rate} Hz`);
      if (o.channels)    add("cat-format",   ICONS.sliders, "Channels", o.channels);
      if (o.audio_format)add("cat-format",   ICONS.save,    "Format", String(o.audio_format).toUpperCase());
      if (seg.output_unit && seg.output_unit !== "original_files")
                         add("cat-segment",  ICONS.layers,  "Output unit", seg.output_unit.replace(/_/g, " "));
      if (seg.duration_min_sec != null || seg.duration_max_sec != null) {
        const lo = seg.duration_min_sec != null ? seg.duration_min_sec + "s" : "—";
        const hi = seg.duration_max_sec != null ? seg.duration_max_sec + "s" : "—";
        add("cat-segment", ICONS.layers, "Clip length", `${lo} → ${hi}`);
      }
      if (seg.speech_policy && seg.speech_policy !== "off")
                         add("cat-vad",      ICONS.scissors,"Speech policy", seg.speech_policy);
      if (q.mos && q.mos !== "off")
                         add("cat-quality",  ICONS.gauge,   "UTMOS", q.mos_threshold != null ? `${q.mos} ≥ ${q.mos_threshold}` : q.mos);
      if (q.sigmos && q.sigmos !== "off") {
        const axes = Array.isArray(q.sigmos_axes) && q.sigmos_axes.length ? q.sigmos_axes.join(",") : "all";
        add("cat-quality", ICONS.gauge, "SIGMOS", `${q.sigmos} (${axes})`);
      }
      if (q.band && q.band !== "off")
                         add("cat-quality",  ICONS.gauge,   "Band", q.band_value || q.band);
      if (sp.mode && sp.mode !== "off") {
        let bits = sp.mode;
        if (sp.target_count != null) bits += ` ·=${sp.target_count}`;
        if (sp.min_count != null)    bits += ` ·≥${sp.min_count}`;
        if (sp.max_count != null)    bits += ` ·≤${sp.max_count}`;
        add("cat-speaker", ICONS.users, "Speakers", bits);
      }
      if (tx.transcript_source && tx.transcript_source !== "off")
                         add("cat-asr",      ICONS.mic,     "Transcripts", tx.transcript_source + (tx.asr_backend ? ` · ${tx.asr_backend}` : ""));
      if (tx.word_timing)add("cat-asr",      ICONS.target,  "Word timing", "on");
      if (tx.wer_mode && tx.wer_mode !== "off")
                         add("cat-quality",  ICONS.gauge,   "WER", tx.wer_max != null ? `${tx.wer_mode} ≤ ${tx.wer_max}` : tx.wer_mode);
      return out;
    }
    function execPillsHtml(data) {
      const exec = data.executor_config || {};
      const cluster = data.cluster || {};
      const mode = exec.execution_mode || "?";
      const findingsErr = (data.findings || []).filter(f => f.severity === "error").length;
      const findingsWarn = (data.findings || []).filter(f => f.severity === "warning").length;
      const pills = [];
      pills.push(`<div class="pipe-pill green">${ICONS.bolt}${escapeHtml(mode)} mode</div>`);
      pills.push(`<div class="pipe-pill">${ICONS.route}${escapeHtml(exec.backend || "auto")}</div>`);
      pills.push(`<div class="pipe-pill">${ICONS.cpu}${escapeHtml(String(cluster.cpus || "?"))} CPU</div>`);
      if (cluster.gpus != null)
        pills.push(`<div class="pipe-pill">${ICONS.bolt}${escapeHtml(String(cluster.gpus))} GPU · ${escapeHtml(String(cluster.gpu_memory_gb || "?"))} GB</div>`);
      if (findingsErr)
        pills.push(`<div class="pipe-pill red">${findingsErr} error${findingsErr === 1 ? "" : "s"}</div>`);
      else if (findingsWarn)
        pills.push(`<div class="pipe-pill amber">${findingsWarn} warning${findingsWarn === 1 ? "" : "s"}</div>`);
      else
        pills.push(`<div class="pipe-pill green">${ICONS.check}validated</div>`);
      return pills.join("");
    }
    function renderPipelineGraph(data) {
      const items = buildFlowItems(data);
      if (items.length) items[items.length - 1].__sink = true;
      const cards = items.map((it, i) => flowCardHtml(it, i)).join("");

      const highlights = intentHighlights(data.intent);
      const hlHtml = highlights.length
        ? `<div class="pipe-highlights">${highlights.map(h => `
            <span class="pipe-hl">
              <span class="hl-icon">${h.icon}</span>
              <span class="hl-label">${escapeHtml(h.label)}</span>
              <span class="hl-val">${escapeHtml(String(h.val))}</span>
            </span>`).join("")}</div>`
        : `<p class="hint" style="margin:0">Defaults only — no overrides from your answers.</p>`;

      const reasons = (data.executor_config && data.executor_config.tuner_reasons) || [];
      const findings = data.findings || [];
      const reasonLines = [
        ...reasons.map(r => r),
        ...findings.map(f => `${(f.severity || "info").toUpperCase()}: ${f.detail}`),
      ];
      const reasoningText = reasonLines.length
        ? reasonLines.join("\n")
        : "Tuner had nothing to say — defaults were used everywhere.";

      const paths = data.paths || {};
      const pathsHtml = `
        <div class="pipe-paths">
          <strong>Run artefacts</strong>: ${escapeHtml(paths.out_dir || "—")}<br/>
          compiled.yaml · ir.validated.json · findings.json · dry_run.json · intent.json
        </div>`;

      const yamlEsc = escapeHtml(data.yaml || "");
      const intentEsc = escapeHtml(JSON.stringify(data.intent || {}, null, 2));
      const tunerEsc = escapeHtml(JSON.stringify(data.tuner || [], null, 2));
      const reasoningEsc = escapeHtml(reasoningText);

      formArea.innerHTML = `
        <div class="pipe-summary" style="animation:flowPop .4s ease both">
          <div>
            <h2>${ICONS.check} Pipeline ready</h2>
            <div class="sub">${escapeHtml(data.message || "Compiled and dry-run validated.")}</div>
          </div>
          <div class="pipe-pills">${execPillsHtml(data)}</div>
        </div>

        <div class="flow-wrap" id="flowWrap">
          <svg class="flow-svg" id="flowSvg" aria-hidden="true"></svg>
          <div class="flow-grid" id="flowGrid">${cards}</div>
        </div>

        <div class="pipe-panel">
          <h3>${ICONS.target} What we tuned for you</h3>
          ${hlHtml}
        </div>

        ${pathsHtml}

        <details class="pipe-advanced" id="pipeAdvanced">
          <summary>
            <span class="chev">▶</span>
            For power users · view raw config (YAML, intent, tuner, reasoning)
          </summary>
          <div class="adv-body">
            <div class="adv-tabs">
              <button class="adv-tab active" data-tab="yaml">YAML</button>
              <button class="adv-tab" data-tab="intent">Intent JSON</button>
              <button class="adv-tab" data-tab="tuner">Tuner JSON</button>
              <button class="adv-tab" data-tab="reasoning">Reasoning</button>
            </div>
            <pre id="advYaml">${yamlEsc}</pre>
            <pre id="advIntent" style="display:none">${intentEsc}</pre>
            <pre id="advTuner" style="display:none">${tunerEsc}</pre>
            <pre id="advReasoning" style="display:none">${reasoningEsc}</pre>
          </div>
        </details>

        <div class="actions-row" style="margin-top:16px">
          <button class="secondary" id="resetAfterBuild">&#8592; New prompt</button>
        </div>`;

      document.getElementById("resetAfterBuild").addEventListener("click", resetAll);
      document.querySelectorAll(".adv-tab").forEach(btn => {
        btn.addEventListener("click", () => {
          document.querySelectorAll(".adv-tab").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          const target = btn.dataset.tab;
          document.getElementById("advYaml").style.display      = target === "yaml"      ? "" : "none";
          document.getElementById("advIntent").style.display    = target === "intent"    ? "" : "none";
          document.getElementById("advTuner").style.display     = target === "tuner"     ? "" : "none";
          document.getElementById("advReasoning").style.display = target === "reasoning" ? "" : "none";
        });
      });

      // draw + redraw on resize / scroll
      requestAnimationFrame(() => drawFlowConnectors());
      const wrap = document.getElementById("flowWrap");
      if (wrap && "ResizeObserver" in window) {
        const ro = new ResizeObserver(() => drawFlowConnectors());
        ro.observe(wrap);
      }
      if (wrap) wrap.addEventListener("scroll", drawFlowConnectors);
      window.addEventListener("resize", drawFlowConnectors);
    }
    function drawFlowConnectors() {
      const svg = document.getElementById("flowSvg");
      const wrap = document.getElementById("flowWrap");
      const grid = document.getElementById("flowGrid");
      if (!svg || !wrap || !grid) return;
      const cards = grid.querySelectorAll(".flow-card");
      const cardsArr = Array.from(cards);
      if (cardsArr.length < 2) { svg.innerHTML = ""; return; }
      // size svg to scroll area
      const fullW = Math.max(grid.scrollWidth, wrap.clientWidth);
      const fullH = wrap.clientHeight;
      svg.setAttribute("viewBox", `0 0 ${fullW} ${fullH}`);
      svg.setAttribute("width", fullW);
      svg.setAttribute("height", fullH);
      svg.style.width = fullW + "px";
      svg.style.height = fullH + "px";
      const wrapRect = wrap.getBoundingClientRect();
      // resolve per-category stroke from CSS variable
      const css = getComputedStyle(document.documentElement);
      const paths = [];
      for (let i = 0; i < cardsArr.length - 1; i++) {
        const a = cardsArr[i].getBoundingClientRect();
        const b = cardsArr[i + 1].getBoundingClientRect();
        const x1 = a.right - wrapRect.left + wrap.scrollLeft;
        const y1 = a.top - wrapRect.top + a.height / 2;
        const x2 = b.left - wrapRect.left + wrap.scrollLeft;
        const y2 = b.top - wrapRect.top + b.height / 2;
        const dx = Math.max(40, (x2 - x1) * 0.55);
        const path = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
        const nextColor = (getComputedStyle(document.documentElement)
          .getPropertyValue("--nv-green-dark").trim()) || "#4E7A00";
        paths.push(`
          <g style="color:${nextColor}">
            <circle class="endpoint" cx="${x1}" cy="${y1}"></circle>
            <path class="connector" d="${path}" stroke="currentColor"></path>
            <circle class="endpoint" cx="${x2}" cy="${y2}"></circle>
          </g>`);
      }
      svg.innerHTML = paths.join("");
    }
    function handleBuildResponse(data) {
      if (data.status === "ready") {
        setDone("Pipeline ready");
        renderPipelineGraph(data);
      } else {
        setError("Compilation failed");
        formArea.innerHTML = `<div class="section warning-box">${escapeHtml(data.error || "Unknown error")}</div>`;
      }
    }
    function formatResources(res) {
      if (!res || Object.keys(res).length === 0) return "card-default";
      const parts = [];
      if (res.cpus != null) parts.push(`cpus=${res.cpus}`);
      if (res.gpus != null) parts.push(`gpus=${res.gpus}`);
      if (res.gpu_memory_gb != null) parts.push(`gpu_mem=${res.gpu_memory_gb}`);
      return parts.join(" ");
    }
    function formatHints(hints) {
      if (!hints || Object.keys(hints).length === 0) return "auto (backend)";
      const parts = [];
      for (const k of Object.keys(hints)) {
        parts.push(`${k}=${hints[k]}`);
      }
      return parts.join(" ");
    }
    function resetAll() {
      sessionId = null; currentForm = null;
      pendingAnswers = {}; pendingFreeform = {};
      revealedFollowups = {}; pickedOptions = {};
      setIdle();
      formArea.innerHTML = `
        <div class="welcome">
          <div class="welcome-icon">&#127911;</div>
          <h2>Build an audio pipeline</h2>
          <p>Describe your audio curation goal in the sidebar and press <strong>Plan pipeline</strong>.</p>
        </div>`;
    }
    document.getElementById("plan").addEventListener("click", planFlow);
    document.getElementById("reset").addEventListener("click", resetAll);
    document.getElementById("dataset").value = defaults.dataset || "";
    document.getElementById("kind").value = defaults.kind || "directory";
    document.getElementById("out").value = defaults.out || "";
    document.getElementById("model").value = defaults.model || "";
    document.getElementById("cluster_cpus").value = defaults.cluster_cpus || 8;
    document.getElementById("cluster_gpus").value = defaults.cluster_gpus != null ? defaults.cluster_gpus : 0;
    document.getElementById("cluster_gpu_memory_gb").value = defaults.cluster_gpu_memory_gb || 24;

    /* ── login wiring ──────────────────────────────────────────── */
    document.getElementById("loginForm").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const btn   = document.getElementById("loginBtn");
      const email = document.getElementById("loginEmail").value.trim();
      const key   = document.getElementById("loginKey").value.trim();
      const err   = document.getElementById("loginError");
      err.textContent = "";
      if (!email || !key) { err.textContent = "Email and API key are required."; return; }
      btn.disabled = true; btn.textContent = "Signing in…";
      try {
        await tryLogin(email, key);
      } catch (e) {
        err.textContent = e.message || "Sign-in failed";
      } finally {
        btn.disabled = false; btn.textContent = "Continue";
      }
    });
    document.getElementById("logoutBtn").addEventListener("click", doLogout);
    const stored = loadStoredAuth();
    if (stored && stored.user_id) {
      userId = stored.user_id; userEmail = stored.email || "";
      showAuthChip(); hideLogin();
    } else {
      showLogin("");
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

