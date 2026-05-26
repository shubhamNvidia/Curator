#!/usr/bin/env bash
# Restart the curator-adv agentic web UI (`python -m nemo_curator.agentic.web`).
#
# Step 1: kill any process already listening on the chosen port.
# Step 2: launch a fresh instance, detached, with unbuffered logs.
#
# By default the server now binds to 0.0.0.0 so other machines on the
# cluster can reach it. Each user signs in with their own NGC / build
# API key (the login page is shown first); per-user activity logs land
# in WEBSITE_DIR/sessions/<email>__<userid>.log.
#
# Usage:
#   scripts/restart_web.sh                       # defaults: 0.0.0.0:7860
#   HOST=127.0.0.1 PORT=7860 scripts/restart_web.sh
#   scripts/restart_web.sh --status              # just report current state
#   scripts/restart_web.sh --stop                # kill only, don't relaunch
#
# Env vars honored:
#   HOST              bind host                (default 0.0.0.0)
#   PORT              bind port                (default 7860)
#   LOG_FILE          server log destination   (default WEBSITE_DIR/server.log)
#   WEBSITE_DIR       per-user log root        (default <repo>/../website)
#   NVIDIA_API_KEY    server-side fallback key (each user supplies their own
#                                               at login; this is only used
#                                               if a code path ever needs an
#                                               anonymous client)
#   CURATOR_ROOT      project root             (default: repo containing this script)
#   LLM_DEBUG         1 = log every LLM prompt + completion to LOG_FILE
#                     (sets OPENAI_LOG=debug, LOGURU_LEVEL=DEBUG, PYTHONLOGLEVEL=DEBUG)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_ROOT="${CURATOR_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$CURATOR_ROOT/.venv}"
WEBSITE_DIR="${WEBSITE_DIR:-$(cd "$CURATOR_ROOT/.." && pwd)/website}"
mkdir -p "$WEBSITE_DIR/sessions"
export CURATOR_ADV_WEBSITE_DIR="$WEBSITE_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
LOG_FILE="${LOG_FILE:-$WEBSITE_DIR/server.log}"

# Default LLM credentials. Override by exporting NVIDIA_API_KEY before running.
# NOTE: per-user keys collected at login take precedence; this is only a
# safety net for any future read-only / anonymous code path.
: "${NVIDIA_API_KEY:=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}"
export NVIDIA_API_KEY

# Verbose LLM I/O logging.
#   LLM_DEBUG=1 ./scripts/restart_web.sh
#     -> openai SDK logs full request body (`messages=...`) and full response
#        body (`choices=...`) to stderr, which lands in LOG_FILE. Auth header
#        is redacted by the SDK. Loguru is bumped to DEBUG so llm.chat() also
#        prints model/tier/timeout metadata for every call.
LLM_DEBUG="${LLM_DEBUG:-0}"
if [[ "$LLM_DEBUG" == "1" || "$LLM_DEBUG" == "true" || "$LLM_DEBUG" == "yes" ]]; then
  export OPENAI_LOG="${OPENAI_LOG:-debug}"
  export LOGURU_LEVEL="${LOGURU_LEVEL:-DEBUG}"
  export PYTHONLOGLEVEL="${PYTHONLOGLEVEL:-DEBUG}"
  LLM_DEBUG_ON=1
else
  LLM_DEBUG_ON=0
fi

MODE="restart"
for arg in "$@"; do
  case "$arg" in
    --stop)    MODE="stop" ;;
    --status)  MODE="status" ;;
    --help|-h) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[restart_web] %s\n' "$*"; }

find_listener_pids() {
  ss -tlnpH 2>/dev/null \
    | awk -v p=":$PORT" '$4 ~ p"$"{print $0}' \
    | grep -oE 'pid=[0-9]+' \
    | cut -d= -f2 \
    | sort -u
}

stop_existing() {
  local pids
  pids="$(find_listener_pids || true)"
  if [[ -z "$pids" ]]; then
    log "no existing process on :$PORT"
    return 0
  fi
  log "stopping existing pid(s) on :$PORT: $(echo $pids | tr '\n' ' ')"
  for pid in $pids; do
    kill "$pid" 2>/dev/null || true
  done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    [[ -z "$(find_listener_pids || true)" ]] && break
  done
  pids="$(find_listener_pids || true)"
  if [[ -n "$pids" ]]; then
    log "force-killing: $(echo $pids | tr '\n' ' ')"
    for pid in $pids; do
      kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
  fi
  if [[ -n "$(find_listener_pids || true)" ]]; then
    log "ERROR: port :$PORT is still in use after kill" >&2
    return 1
  fi
  log "port :$PORT free"
}

status() {
  local pids
  pids="$(find_listener_pids || true)"
  if [[ -z "$pids" ]]; then
    log "no process listening on $HOST:$PORT"
    return 1
  fi
  log "listener pid(s) on :$PORT: $(echo $pids | tr '\n' ' ')"
  ps -o pid,etime,stat,rss,cmd -p $pids 2>/dev/null || true
  curl -s -o /dev/null -w "[restart_web] HTTP %{http_code} | %{size_download} bytes | %{time_total}s\n" \
    --max-time 5 "http://$HOST:$PORT/" || true
}

launch() {
  if [[ ! -d "$CURATOR_ROOT/nemo_curator/agentic" ]]; then
    log "ERROR: cannot find nemo_curator/agentic under CURATOR_ROOT=$CURATOR_ROOT" >&2
    return 1
  fi
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    log "ERROR: venv not found at $VENV_DIR" >&2
    return 1
  fi
  if [[ -z "${NVIDIA_API_KEY:-}" && -z "${CURATOR_ADV_LLM_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    log "WARNING: no LLM API key in env (NVIDIA_API_KEY / CURATOR_ADV_LLM_API_KEY / OPENAI_API_KEY); LLM calls will 401"
  fi

  cd "$CURATOR_ROOT"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  : > "$LOG_FILE"
  log "launching: python -u -m nemo_curator.agentic.web --host $HOST --port $PORT"
  log "server log:  $LOG_FILE"
  log "website dir: $WEBSITE_DIR (per-user logs under sessions/)"
  if [[ "$LLM_DEBUG_ON" == "1" ]]; then
    log "LLM_DEBUG=1 -> OPENAI_LOG=$OPENAI_LOG, LOGURU_LEVEL=$LOGURU_LEVEL (prompts + completions will be logged)"
  fi
  PYTHONUNBUFFERED=1 nohup python -u -m nemo_curator.agentic.web \
    --host "$HOST" --port "$PORT" \
    > "$LOG_FILE" 2>&1 &
  local new_pid=$!
  disown 2>/dev/null || true
  log "spawned pid $new_pid; waiting for port :$PORT ..."

  for i in $(seq 1 30); do
    sleep 1
    if find_listener_pids | grep -q .; then
      log "listening after ${i}s"
      break
    fi
    if ! kill -0 "$new_pid" 2>/dev/null; then
      log "ERROR: process $new_pid died before binding; see $LOG_FILE" >&2
      tail -n 40 "$LOG_FILE" >&2 || true
      return 1
    fi
  done

  if ! find_listener_pids | grep -q .; then
    log "ERROR: process $new_pid alive but not listening on :$PORT after 30s" >&2
    tail -n 40 "$LOG_FILE" >&2 || true
    return 1
  fi

  status || true
  log "URL: http://$HOST:$PORT/"
}

case "$MODE" in
  status)  status ;;
  stop)    stop_existing ;;
  restart) stop_existing && launch ;;
esac
