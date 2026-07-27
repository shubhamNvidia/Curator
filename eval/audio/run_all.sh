#!/usr/bin/env bash
# Orchestrate the audio-agent test suite (see AGENT_TEST_PLAN.md 11-12).
#   deterministic eval + card gate -> level-14 generalization -> trace/judge
#   selftests -> regression snapshot -> (opt-in) LLM plane -> (opt-in) E2E -> final report.
# Usage:
#   bash eval/audio/run_all.sh                 # deterministic planes only
#   bash eval/audio/run_all.sh --llm           # + SDK-driven LLM-plane (needs CURSOR_API_KEY)
#   bash eval/audio/run_all.sh --e2e           # + GPU/Ray E2E on /tmp/aa_real
#   bash eval/audio/run_all.sh --sim           # + end-to-end user-simulation personas (needs CURSOR_API_KEY + GPU)
#   bash eval/audio/run_all.sh --llm --e2e --sim   # everything
# Env: MODEL / MODEL_SUT / MODEL_USER (SDK models), RAY_ADDRESS, AUDIO_AGENT_WORKSPACE.
set -u
cd "$(dirname "$0")/../.." || exit 2          # repo root
PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:-claude-opus-4-8}"
DO_LLM=0; DO_E2E=0; DO_SIM=0
for a in "$@"; do
  case "$a" in
    --llm) DO_LLM=1 ;;
    --e2e) DO_E2E=1 ;;
    --sim) DO_SIM=1 ;;
    *) echo "unknown arg: $a" ;;
  esac
done
mkdir -p eval/audio/reports
rc=0

echo "== [1] deterministic eval + card gate =="
$PY -m eval.audio.run_eval --report eval/audio/reports/latest.json || rc=1

echo "== [2] level-14 generalization (novel stage + card) =="
$PY eval/audio/fixtures/novel_stage/novel_card_check.py || rc=1

echo "== [3] trace_check selftest =="
$PY -m eval.audio.trace_check --selftest || rc=1

echo "== [4] judge selftest =="
$PY -m eval.audio.judge --selftest || rc=1

echo "== [5] regression snapshot =="
$PY -m eval.audio.snapshot --check \
  || echo "(snapshot missing or drifted; run '$PY -m eval.audio.snapshot --write' to (re)baseline)"

if [ "$DO_LLM" = "1" ]; then
  echo "== [6] LLM plane: capture traces ($MODEL) + aggregate =="
  if [ -z "${CURSOR_API_KEY:-}" ]; then
    echo "  CURSOR_API_KEY not set - skipping SDK capture (grading existing traces only)"
  else
    $PY -m eval.audio.agent_runner --batch --mode sdk --model "$MODEL" || true
  fi
  $PY -m eval.audio.aggregate_traces --report eval/audio/reports/llm_plane.json || rc=1
else
  echo "== [6] LLM plane: skipped (pass --llm) =="
fi

if [ "$DO_E2E" = "1" ]; then
  echo "== [7] GPU E2E on ${AUDIO_AGENT_WORKSPACE:-/tmp/aa_real} =="
  $PY -m eval.audio.run_e2e --report eval/audio/reports/e2e.json || rc=1
else
  echo "== [7] GPU E2E: skipped (pass --e2e) =="
fi

if [ "$DO_SIM" = "1" ]; then
  echo "== [7b] end-to-end user-simulation personas (SUT=${MODEL_SUT:-claude-opus-4-8}, USER=${MODEL_USER:-composer-2.5}) =="
  if [ -z "${CURSOR_API_KEY:-}" ]; then
    echo "  CURSOR_API_KEY not set - skipping user-simulation"
  else
    $PY -m eval.audio.simulate_user --report eval/audio/reports/e2e_from_prompt.json || rc=1
  fi
else
  echo "== [7b] user-simulation: skipped (pass --sim) =="
fi

echo "== [8] inside-out debug report + merged final report =="
$PY -m eval.audio.debug_report || rc=1
$PY -m eval.audio.final_report || rc=1

echo "== done (rc=$rc) - reports in eval/audio/reports/ =="
exit $rc
