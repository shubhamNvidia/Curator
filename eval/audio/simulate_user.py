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

"""End-to-end user-simulation harness (AGENT_TEST_PLAN.md).

Two SDK agents per persona:
  * SUT  - the curation agent (MODEL_SUT, default claude-opus-4-8) with the
           audio-agent MCP tools + SKILL preamble.
  * USER - a second LLM (MODEL_USER) that plays the user: opens with the persona's
           prompt and answers the SUT's clarifying questions from persona_facts.
Then the agent's OWN recipe is executed on real FLEURS (smoke -> run -> report ->
verify) and graded (success kind, validity, module set, and composition ordering
from patterns/composition.yaml). Every gap/bug is logged as a finding (fix later).

    export CURSOR_API_KEY=... RAY_ADDRESS=127.0.0.1:6457 AUDIO_AGENT_WORKSPACE=/tmp/aa_real
    python -m eval.audio.simulate_user --ids L01_duration          # one persona
    python -m eval.audio.simulate_user --group pattern             # a group
    python -m eval.audio.simulate_user                             # all personas
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re

from eval.audio import agent_runner as ar

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = ar._ROOT
_PERSONAS = os.path.join(_HERE, "personas.yaml")
_COMPOSITION = os.path.join(_ROOT, "nemo_curator", "audio_agent", "knowledge", "patterns", "composition.yaml")
_SIM_TRACES = os.path.join(_HERE, "traces", "sim")
_REPORT = os.path.join(_HERE, "reports", "e2e_from_prompt.json")
_FINDINGS = os.path.join(_HERE, "reports", "findings.json")

_MODEL_SUT = os.environ.get("MODEL_SUT", "claude-opus-4-8")
_MODEL_USER = os.environ.get("MODEL_USER", "composer-2.5")
_MAX_TURNS = int(os.environ.get("SIM_MAX_TURNS", "6"))
_AA_REAL = os.environ.get("AUDIO_AGENT_WORKSPACE", "/tmp/aa_real")
_MANIFEST = os.path.join(_AA_REAL, "manifest.jsonl")
_PATH_HINTS = ("path", "dir", "manifest", "file_paths", "raw_data_dir", "output")


def _load_yaml(path):
    import yaml

    return yaml.safe_load(open(path, encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The two-agent conversation
# --------------------------------------------------------------------------- #

def _is_question(text: str) -> bool:
    return "?" in (text or "")


def _user_reply(persona: dict, transcript: list[dict], sut_text: str, key: str | None) -> str:
    """One user turn from the USER LLM (MODEL_USER), in the persona's voice."""
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

    convo = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in transcript if t.get("text"))
    prompt = (
        "You are role-playing a USER talking to an audio data-curation assistant. Stay in "
        "character as a non-expert user - use plain language, never mention internal stage "
        "names or parameters.\n"
        f"YOUR GOAL: {persona.get('goal')}\n"
        f"FACTS ABOUT YOUR DATA (reveal only what's relevant when asked): {persona.get('persona_facts')}\n"
        f"YOUR PREFERENCES (use these to answer the assistant's questions): {persona.get('preferences')}\n\n"
        f"CONVERSATION SO FAR:\n{convo}\n\n"
        f"The assistant just said:\n{sut_text}\n\n"
        "Reply with ONLY your next message as the user, in 1-2 short sentences. If the assistant "
        "asked a question, answer it from your preferences/facts. If the assistant has proposed a "
        "plan that meets your goal, approve it briefly (e.g. 'that sounds good, go ahead'). If it "
        "says your request can't be done, acknowledge and ask what else is possible. No preamble, "
        "no quotes, no tool use."
    )
    try:
        res = Agent.prompt(prompt, AgentOptions(model=_MODEL_USER, api_key=key, local=LocalAgentOptions(cwd=_ROOT)))
        txt = (getattr(res, "result", None) or "").strip()
        return txt or "That sounds good, go ahead."
    except Exception as e:  # noqa: BLE001 - user-sim failure shouldn't abort the persona
        return f"(user-sim error: {type(e).__name__}) go with your best judgement."


def run_conversation(persona: dict, *, model_sut: str, key: str | None) -> dict:
    """Drive the multi-turn SUT<->USER loop; return transcript + final recipe + outcome."""
    from cursor_sdk import (
        Agent,
        AgentOptions,
        CursorAgentError,
        LocalAgentOptions,
        SDKToolUseMessage,
        StdioMcpServerConfig,
    )

    mcp = {
        "audio-agent": StdioMcpServerConfig(
            command=os.path.join(_ROOT, ".venv", "bin", "python"),
            args=["-m", "nemo_curator.audio_agent.mcp_server"],
            cwd=_ROOT, env=dict(os.environ),
        )
    }
    opts = AgentOptions(model=model_sut, api_key=key, local=LocalAgentOptions(cwd=_ROOT), mcp_servers=mcp)

    transcript: list[dict] = []
    all_tool_calls: list[dict] = []
    recipe = None
    outcome = "no_reply"
    asked_clarification = False

    try:
        with Agent.create(opts) as sut:
            user_msg = persona["opening_prompt"]
            transcript.append({"turn": 0, "speaker": "user", "text": user_msg})
            for turn in range(1, _MAX_TURNS + 1):
                send = (ar._preamble() + "\n\nUSER REQUEST: " + user_msg) if turn == 1 else user_msg
                run = sut.send(send)
                calls = []
                for msg in run.messages():
                    if isinstance(msg, SDKToolUseMessage) and getattr(msg, "result", None) is not None:
                        verb, targs, tres = ar._extract_mcp(msg)
                        calls.append({"verb": verb, "args": targs, "result": tres})
                run.wait()
                sut_text = ""
                try:
                    sut_text = run.text() or ""
                except Exception:  # noqa: BLE001
                    sut_text = ""
                all_tool_calls.extend(calls)
                transcript.append({"turn": turn, "speaker": "agent", "text": sut_text, "tool_calls": calls})

                recipe = ar._parse_recipe(sut_text)
                if recipe:
                    outcome = "recipe"
                    break
                if _is_question(sut_text):
                    asked_clarification = True
                    user_msg = _user_reply(persona, transcript, sut_text, key)
                    transcript.append({"turn": turn, "speaker": "user", "text": user_msg})
                    continue
                # a statement with no recipe and no question -> refusal / dead-end
                outcome = "refused" if sut_text else "no_reply"
                break
            else:
                outcome = "clarify_loop"  # hit the turn cap still asking
    except CursorAgentError as e:
        return {"blocked": True, "error": f"startup failed: {e}", "transcript": transcript,
                "tool_calls": all_tool_calls, "final_recipe": None, "outcome": "blocked",
                "asked_clarification": asked_clarification}

    return {"blocked": False, "transcript": transcript, "tool_calls": all_tool_calls,
            "final_recipe": recipe, "outcome": outcome, "asked_clarification": asked_clarification}


# --------------------------------------------------------------------------- #
# Execute the agent's OWN recipe on real data
# --------------------------------------------------------------------------- #

def _substitute_paths(recipe: dict, out_dir: str) -> dict:
    """Replace placeholder path params with the real manifest / an output dir."""
    import copy

    rec = copy.deepcopy(recipe)
    os.makedirs(out_dir, exist_ok=True)
    for st in rec.get("stages", []):
        params = st.get("params") or {}
        for k, v in list(params.items()):
            if not isinstance(v, str):
                continue
            kl = k.lower()
            if not any(h in kl for h in _PATH_HINTS):
                continue
            if "manifest" in kl or ("path" in kl and "output" not in kl and "out" not in kl):
                params[k] = _MANIFEST
            elif "dir" in kl:
                params[k] = out_dir
            else:  # output_path / *_path outputs
                params[k] = os.path.join(out_dir, "out.jsonl")
    return rec


def _ensure_ray_headroom(force: bool = True, threshold: float = 0.6) -> None:
    """Reset the harness's Ray cluster before executing a persona (OPT-IN).

    Two runtime-failure modes were traced to a long-lived cluster degrading across many
    sequential heavy GPU runs: (1) host-RAM OOM -- 26 model runs leaked ~69 GB of
    StageWorker RAM (L08); and (2) accumulated dead/unhealthy workers -> ``executor_error``
    even when RAM is moderate (L08/Cx3 re-run). A fresh cluster per executable persona
    eliminates both.

    Restarting Ray and reclaiming orphaned temp dirs is DESTRUCTIVE on a shared host, so
    it is disabled unless ``AUDIO_AGENT_SIM_RESET_RAY`` is truthy -- set it only on a box
    you own for the full-campaign run. Even when enabled, temp reclamation is scoped to
    directories owned by the current user (never another user's files), and ``ray stop``
    only affects this user's own Ray processes. ``force=False`` restarts only when RAM
    crosses ``threshold``. Best-effort: never aborts a persona.
    """
    if os.environ.get("AUDIO_AGENT_SIM_RESET_RAY", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return  # safe default: no destructive Ray reset / temp cleanup on shared hosts

    import subprocess
    import time

    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {ln.split(":")[0]: int(ln.split()[1]) for ln in f if ":" in ln}
        total, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
        used_frac = 1 - (avail / total) if total else 0.0
        if not force and used_frac < threshold:
            return
        ray = os.path.join(_ROOT, ".venv", "bin", "ray")
        port = os.environ.get("RAY_ADDRESS", "127.0.0.1:6457").rsplit(":", 1)[-1]
        subprocess.run([ray, "stop"], capture_output=True, timeout=60, check=False)
        time.sleep(3)
        __import__("shutil").rmtree("/tmp/ray_aa", ignore_errors=True)
        subprocess.run(
            [ray, "start", "--head", "--port", port, "--temp-dir", "/tmp/ray_aa",
             "--plasma-directory", "/tmp", "--num-gpus", "1", "--num-cpus", "16",
             "--disable-usage-stats", "--dashboard-host", "127.0.0.1"],
            capture_output=True, timeout=120, check=False,
            env=dict(os.environ, RAY_MAX_LIMIT_FROM_API_SERVER="40000"),
        )
        # Reclaim orphaned NeMo model-unpack temp dirs (~2.4 GB each). They leak when a
        # worker is killed mid-unpack (the OOM path) and, unbounded, fill the disk ->
        # OSError Errno 28 surfacing as executor_error. Safe here: no pipeline is running.
        import glob
        import shutil as _shutil

        freed = 0
        my_uid = os.getuid()
        for d in glob.glob("/tmp/tmp*"):
            try:
                # Only reclaim OUR OWN orphaned unpack dirs -- never delete another
                # user's temp files, even if they match the NeMo model signature.
                if (
                    os.path.isdir(d)
                    and os.stat(d).st_uid == my_uid
                    and any(
                        os.path.exists(os.path.join(d, f))
                        for f in ("model_weights.ckpt", "model_config.yaml")
                    )
                ):
                    _shutil.rmtree(d, ignore_errors=True)
                    freed += 1
            except OSError:
                pass
        time.sleep(5)
        print(
            f"[sim] reset Ray before execute (host RAM was {used_frac:.0%}; cleared {freed} orphaned model temps)",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 - headroom management must never abort a persona
        print(f"[sim] ray headroom check skipped: {e}", flush=True)


def _execute(recipe: dict, persona_id: str) -> dict:
    from nemo_curator import audio_agent as aa

    _ensure_ray_headroom()  # opt-in (AUDIO_AGENT_SIM_RESET_RAY): curb worker-RAM accumulation across runs
    out_dir = os.path.join(_AA_REAL, "sim_out", persona_id)
    rec = _substitute_paths(recipe, out_dir)
    row: dict = {"attempted": True}
    try:
        sm = aa.smoke(rec, sample=4, data=_MANIFEST)
        if sm.get("status") == "refused":
            return {**row, "status": "smoke_refused", "reason": sm.get("reason")}
        row["smoke"] = {"ran": sm.get("ran"), "retained": sm.get("retained"), "errors": len(sm.get("errors") or [])}
        rr = aa.run(rec, confirm=sm.get("config_hash"), data=_MANIFEST, smoke_token=sm.get("smoke_token"))
        row["status"] = rr.get("status")
        rep = rr.get("report") or {}
        row["accepted"] = rep.get("accepted")
        row["input_count"] = rep.get("input_count")
        row["failures"] = [f.get("code") for f in (rep.get("failure_reasons") or [])]
        try:
            from eval.audio.run_e2e import _metrics_summary

            row["metrics"] = _metrics_summary(rep)
        except Exception:  # noqa: BLE001
            row["metrics"] = {}
        ev = {"retained": rep.get("accepted") or 0, "input_count": rep.get("input_count") or 0}
        row["verify_overall"] = aa.verify(
            [{"id": "kept", "type": "yield", "kind": "absolute", "severity": "must", "check": {"op": ">", "value": 0}}], ev
        ).get("overall")
    except Exception as e:  # noqa: BLE001 - execution failure is a finding, not a crash
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


# --------------------------------------------------------------------------- #
# Grading (success kind + validity + module set + composition ordering)
# --------------------------------------------------------------------------- #

def _tok_index(refs: list[str], token: str, equiv: dict) -> int | None:
    """Index of a stage token in the recipe, resolving an equivalence-group key to the
    earliest present member (or, for the 'after' side, we still use earliest)."""
    if token in equiv:
        idxs = [refs.index(m) for m in equiv[token] if m in refs]
        return min(idxs) if idxs else None
    return refs.index(token) if token in refs else None


def _composition_kinds() -> dict:
    try:
        pats = _load_yaml(_COMPOSITION).get("patterns", [])
        return {p["id"]: p.get("kind", "advisory") for p in pats}
    except Exception:  # noqa: BLE001
        return {}


def _grade(persona: dict, conv: dict, execution: dict | None) -> dict:  # noqa: C901
    exp = persona.get("expected", {}) or {}
    equiv = exp.get("equivalence", {}) or {}
    recipe = conv.get("final_recipe")
    refs = [s.get("ref") for s in (recipe or {}).get("stages", [])]
    findings: list[dict] = []
    dims: dict[str, dict] = {}
    want_kind = (persona.get("success") or {}).get("kind", "executes")
    comp_kinds = _composition_kinds()

    def finding(stage, category, severity, what, fix):
        findings.append({
            "id": f"{persona['id']}::{stage}", "source": persona["id"], "stage": stage,
            "category": category, "severity": severity, "what_happened": what,
            "evidence": {"recipe": refs, "outcome": conv.get("outcome")},
            "suggested_fix": fix, "status": "open",
        })

    # 1) success kind
    outcome = conv.get("outcome")
    asked = conv.get("asked_clarification")
    if want_kind == "refuses":
        # PASS if the agent DECLINED the original unsupported ask (its first turn was not
        # a recipe) and never used a forbidden stage. A later user-initiated pivot to a
        # supported request that the agent then builds is acceptable (refuse + redirect).
        transcript = conv.get("transcript") or []
        first_agent = next((t for t in transcript if t.get("speaker") == "agent"), None)
        first_had_recipe = bool(ar._parse_recipe(first_agent.get("text", ""))) if first_agent else False
        used_forbidden = any(f in refs for f in (exp.get("forbidden", []) or []))
        ok = (not first_had_recipe) and (not used_forbidden)
        dims["success"] = {"pass": ok, "note": f"declined_original={not first_had_recipe} used_forbidden={used_forbidden} final_outcome={outcome}"}
        if not ok:
            finding("intent", "Unsupported requirement", "P1",
                    "did not decline the unsupported request (built for it immediately or used a forbidden stage)",
                    "refuse/redirect the unsupported ask before building anything")
    elif want_kind == "clarifies":
        dims["success"] = {"pass": bool(asked), "note": f"asked={asked} (wanted clarify)"}
        if not asked:
            finding("clarify", "Missing clarification", "P1", "did not ask on an ambiguous/under-specified request", "ask a user-facing question before building")
    else:  # executes
        dims["success"] = {"pass": bool(recipe), "note": f"outcome={outcome} (wanted a recipe)"}
        if not recipe:
            finding("plan", "Failed recovery", "P1", f"no runnable recipe produced (outcome={outcome})", "build a runnable recipe for a clear request")

    # 2) forbidden modules (always) + required/equivalence (only when a recipe exists)
    for forb in exp.get("forbidden", []) or []:
        if forb in refs:
            dims.setdefault("modules", {"pass": True, "note": ""})
            dims["modules"] = {"pass": False, "note": f"used forbidden {forb}"}
            finding("select", "Incorrect module selection", "P1", f"used forbidden/hallucinated stage {forb}", "do not use this stage for this request")
    if recipe:
        problems = []
        for req in exp.get("required", []) or []:
            if req not in refs:
                problems.append(f"missing {req}")
        for g, members in equiv.items():
            if not (set(members) & set(refs)):
                problems.append(f"no member of '{g}' ({members})")
        mod_ok = not problems and "modules" not in dims or (not problems and dims.get("modules", {}).get("pass", True))
        if problems:
            dims["modules"] = {"pass": False, "note": "; ".join(problems)}
            finding("select", "Incorrect module selection", "P1", "; ".join(problems), "select the capability that produces the requested output")
        else:
            dims.setdefault("modules", {"pass": True, "note": "module set matches"})

        # 3) validity (re-validate the agent's recipe with the real core)
        try:
            from nemo_curator import audio_agent as aa

            v = aa.validate(recipe, expected_outputs=exp.get("output_roles") or None)
            vok = v.get("runnable") is not False and v.get("status") != "fail"
            dims["validity"] = {"pass": vok, "note": f"status={v.get('status')} runnable={v.get('runnable')}"}
            if not vok:
                codes = sorted({i["code"] for pool in ("issues", "card_violations", "gate_flags") for i in v.get(pool, [])})
                finding("validate", "Pipeline validation failure", "P1", f"agent recipe does not validate: {codes}", "fix ordering/keys/params so validate passes")
        except Exception as e:  # noqa: BLE001
            dims["validity"] = {"pass": False, "note": f"validate error: {e}"}

        # 4) order constraints (composition patterns + complex/dependency ordering)
        order_notes = []
        pk = persona.get("pattern_kind", "advisory")
        if persona.get("pattern_id") in comp_kinds:
            pk = comp_kinds[persona["pattern_id"]]
        for before, after in exp.get("order_constraints", []) or []:
            bi = _tok_index(refs, before, equiv)
            ai = _tok_index(refs, after, equiv)
            if bi is None or ai is None:
                order_notes.append(f"{before}->{after}: n/a (a stage absent)")
                continue
            if bi < ai:
                order_notes.append(f"{before}->{after}: ok")
            else:
                order_notes.append(f"{before}->{after}: VIOLATED")
                sev = "P1" if pk == "enforced" else "P2"
                finding("order", "Incorrect module ordering", sev, f"{before} should precede {after} ({pk})", f"reorder so {before} runs before {after}")
        # conditional (diarization-needs-continuous-audio)
        cond = exp.get("conditional_order")
        if cond:
            vad = cond.get("if_before")
            between = cond.get("then_between")
            after_grp = cond.get("and_before")
            ai = _tok_index(refs, after_grp, equiv)
            if vad in refs and ai is not None:
                vi = refs.index(vad)
                if vi < ai:  # VAD fragments BEFORE the diarizer -> a re-join is required between them
                    bi = _tok_index(refs, between, equiv)
                    if bi is None or not (vi < bi < ai):
                        order_notes.append(f"conditional {between} between {vad} and {after_grp}: VIOLATED")
                        finding("order", "Incorrect module ordering", "P1", f"{vad} precedes {after_grp} but {between} is not re-joining in between (diarization needs continuous audio)", f"insert {between} after {vad} and before {after_grp}")
                    else:
                        order_notes.append("conditional re-join: ok")
                else:  # diarizer/separator runs BEFORE any VAD -> continuous audio, no re-join needed
                    order_notes.append(f"{after_grp} runs before {vad} (continuous audio): ok")
        if order_notes:
            dims["ordering"] = {"pass": all("VIOLATED" not in n for n in order_notes if "n/a" not in n), "note": "; ".join(order_notes)}

    # 5) execution (executable personas)
    if execution is not None:
        ex_ok = execution.get("status") == "completed"
        dims["execution"] = {"pass": ex_ok, "note": f"status={execution.get('status')} accepted={execution.get('accepted')}/{execution.get('input_count')}"}
        if not ex_ok:
            finding("execute", "Runtime execution failure", "P1", f"recipe failed to run on real data: {execution.get('error') or execution.get('status')}", "inspect logs; fix the failing stage/params")

    # overall: all HARD dims pass. Advisory ordering deviations are findings, not hard fails.
    hard = {k: d for k, d in dims.items() if not (k == "ordering" and persona.get("pattern_kind") == "advisory")}
    overall = all(d["pass"] for d in hard.values())
    return {"overall": "pass" if overall else "fail", "dimensions": dims, "findings": findings}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_persona(persona: dict, *, key: str | None) -> dict:
    conv = run_conversation(persona, model_sut=_MODEL_SUT, key=key)
    execution = None
    if not conv.get("blocked") and conv.get("final_recipe") and persona.get("executable"):
        execution = _execute(conv["final_recipe"], persona["id"])
    grade = {"overall": "blocked", "dimensions": {}, "findings": []} if conv.get("blocked") else _grade(persona, conv, execution)
    rec = {
        "id": persona["id"], "group": persona.get("group"), "level": persona.get("level"),
        "opening_prompt": persona.get("opening_prompt"), "outcome": conv.get("outcome"),
        "asked_clarification": conv.get("asked_clarification"),
        "final_recipe": [s.get("ref") for s in (conv.get("final_recipe") or {}).get("stages", [])],
        "execution": execution, "grade": grade["overall"], "dimensions": grade["dimensions"],
        "findings": grade["findings"], "blocked": conv.get("blocked", False), "error": conv.get("error"),
    }
    # full trace to disk
    os.makedirs(_SIM_TRACES, exist_ok=True)
    with open(os.path.join(_SIM_TRACES, f"{persona['id']}.json"), "w", encoding="utf-8") as f:
        json.dump({"persona": persona, "conversation": conv, "execution": execution, "grade": grade}, f, indent=2, default=str)
    return rec


def _row_from(persona: dict, conv: dict, execution: dict | None, grade: dict) -> dict:
    return {
        "id": persona["id"], "group": persona.get("group"), "level": persona.get("level"),
        "opening_prompt": persona.get("opening_prompt"), "outcome": conv.get("outcome"),
        "asked_clarification": conv.get("asked_clarification"),
        "final_recipe": [s.get("ref") for s in (conv.get("final_recipe") or {}).get("stages", [])],
        "execution": execution, "grade": grade["overall"], "dimensions": grade["dimensions"],
        "findings": grade["findings"], "blocked": conv.get("blocked", False),
    }


def _emit(rows: list[dict], findings: list[dict], report_path: str) -> dict:
    passed = sum(1 for r in rows if r["grade"] == "pass")
    rep = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model_sut": _MODEL_SUT, "model_user": _MODEL_USER,
        "totals": {"personas": len(rows), "passed": passed, "failed": sum(1 for r in rows if r["grade"] == "fail"),
                   "blocked": sum(1 for r in rows if r["grade"] == "blocked"),
                   "pass_rate": round(passed / max(1, len(rows)), 3)},
        "personas": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str)
    by_sev: dict = {}
    for fnd in findings:
        by_sev[fnd["severity"]] = by_sev.get(fnd["severity"], 0) + 1
    with open(_FINDINGS, "w", encoding="utf-8") as f:
        json.dump({"generated_at": rep["generated_at"], "count": len(findings), "by_severity": by_sev, "findings": findings}, f, indent=2, default=str)
    print(json.dumps(rep["totals"]))
    print(f"[simulate_user] wrote {report_path} + {_FINDINGS} ({len(findings)} findings)")
    return rep


def regrade(report_path: str) -> int:
    """Re-grade the saved traces/sim/*.json against the current grader (no SDK calls)."""
    import glob

    current = {p["id"]: p for p in _load_yaml(_PERSONAS)["personas"]}  # use the up-to-date expectations
    rows, findings = [], []
    for path in sorted(glob.glob(os.path.join(_SIM_TRACES, "*.json"))):
        t = json.load(open(path, encoding="utf-8"))
        saved = t.get("persona") or {}
        persona = current.get(saved.get("id")) or saved  # current persona def over the trace-embedded one
        conv = t.get("conversation") or {}
        execution = t.get("execution")
        if not persona:
            continue
        grade = {"overall": "blocked", "dimensions": {}, "findings": []} if conv.get("blocked") else _grade(persona, conv, execution)
        t["grade"] = grade
        with open(path, "w", encoding="utf-8") as f:
            json.dump(t, f, indent=2, default=str)
        rows.append(_row_from(persona, conv, execution, grade))
        findings.extend(grade["findings"])
        print(f"[regrade] {persona['id']}: {grade['overall']}")
    _emit(rows, findings, report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end user-simulation")
    ap.add_argument("--ids", help="comma-separated persona ids")
    ap.add_argument("--group", choices=["per_level", "complex", "pattern"], help="only this group")
    ap.add_argument("--report", default=_REPORT)
    ap.add_argument("--regrade", action="store_true", help="re-grade saved traces (no SDK calls)")
    args = ap.parse_args(argv)

    if args.regrade:
        return regrade(args.report)

    key = os.environ.get("CURSOR_API_KEY")
    if not key:
        print("[simulate_user] CURSOR_API_KEY not set - cannot drive the SDK agents.")
        return 2

    personas = _load_yaml(_PERSONAS)["personas"]
    if args.ids:
        want = {s.strip() for s in args.ids.split(",")}
        personas = [p for p in personas if p["id"] in want]
    if args.group:
        personas = [p for p in personas if p.get("group") == args.group]

    rows, findings = [], []
    for p in personas:
        print(f"[sim] {p['id']} ...", flush=True)
        r = run_persona(p, key=key)
        rows.append(r)
        findings.extend(r.get("findings") or [])
        print(f"[sim] {p['id']}: grade={r['grade']} outcome={r['outcome']} recipe={r['final_recipe']}"
              + (f" exec={r['execution'].get('status')}" if r.get("execution") else ""))

    _emit(rows, findings, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
