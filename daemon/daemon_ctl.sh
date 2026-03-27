#!/usr/bin/env bash
# daemon_ctl.sh — Manage the ensemble-memory embedding daemon.
#
# Usage: daemon_ctl.sh start|stop|restart|status
#
# The daemon runs a lightweight HTTP server on localhost that keeps the
# sentence-transformers model warm, serving /search and /invalidate_cache.
# This avoids loading the ~80MB model on every user prompt.
#
# Environment variables:
#   ENSEMBLE_MEMORY_PYTHON      Python interpreter (default: python3)
#   ENSEMBLE_MEMORY_DAEMON_PORT Daemon port (default: 9876)
#   ENSEMBLE_MEMORY_DIR         Data dir (default: ~/.ensemble_memory)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DAEMON_PORT="${ENSEMBLE_MEMORY_DAEMON_PORT:-9876}"
ENSEMBLE_MEMORY_DIR="${ENSEMBLE_MEMORY_DIR:-$HOME/.ensemble_memory}"
DAEMON_RUN_DIR="${ENSEMBLE_MEMORY_DIR}/daemon"
PID_FILE="${DAEMON_RUN_DIR}/daemon.pid"
LOG_FILE="${DAEMON_RUN_DIR}/daemon.log"

# ── Locate ensemble-memory project root ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENSEMBLE_MEMORY_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
DAEMON_SCRIPT="${ENSEMBLE_MEMORY_HOME}/daemon/embedding_daemon.py"

# ── Python interpreter ────────────────────────────────────────────────────────
PYTHON3="${ENSEMBLE_MEMORY_PYTHON:-/Users/shane/Documents/playground/.venv/bin/python3}"
if [[ ! -x "$PYTHON3" ]]; then
    PYTHON3="${ENSEMBLE_MEMORY_PYTHON:-$(command -v python3)}"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
_health_check() {
    # Returns 0 if daemon responds to /health, 1 otherwise
    if command -v curl >/dev/null 2>&1; then
        curl -sf --max-time 1 "http://127.0.0.1:${DAEMON_PORT}/health" >/dev/null 2>&1
    else
        "$PYTHON3" -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://127.0.0.1:${DAEMON_PORT}/health', timeout=1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
    fi
}

_read_pid() {
    [[ -f "$PID_FILE" ]] && cat "$PID_FILE" || echo ""
}

_pid_alive() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_start() {
    mkdir -p "$DAEMON_RUN_DIR"

    local pid
    pid="$(_read_pid)"
    if _pid_alive "$pid"; then
        echo "Daemon already running (PID $pid)"
        return 0
    fi

    # Stale PID file — clean up
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"

    if [[ ! -f "$DAEMON_SCRIPT" ]]; then
        echo "ERROR: daemon script not found: $DAEMON_SCRIPT" >&2
        return 1
    fi

    # Start daemon in background, redirect output to log
    ENSEMBLE_MEMORY_DIR="$ENSEMBLE_MEMORY_DIR" \
    ENSEMBLE_MEMORY_DAEMON_PORT="$DAEMON_PORT" \
    ENSEMBLE_MEMORY_HOME="$ENSEMBLE_MEMORY_HOME" \
        nohup "$PYTHON3" "$DAEMON_SCRIPT" \
            >>"$LOG_FILE" 2>&1 &
    local daemon_pid=$!
    echo "$daemon_pid" > "$PID_FILE"

    # Wait up to 10s for /health to respond
    local waited=0
    while [[ $waited -lt 10 ]]; do
        if _health_check; then
            echo "Daemon started on port ${DAEMON_PORT} (PID ${daemon_pid})"
            return 0
        fi
        sleep 1
        (( waited++ ))
    done

    echo "ERROR: Daemon did not become healthy within 10s. Check ${LOG_FILE}" >&2
    return 1
}

cmd_stop() {
    local pid
    pid="$(_read_pid)"

    if [[ -z "$pid" ]]; then
        echo "Daemon not running (no PID file)"
        return 0
    fi

    if ! _pid_alive "$pid"; then
        echo "Daemon not running (stale PID $pid)"
        rm -f "$PID_FILE"
        return 0
    fi

    echo "Stopping daemon (PID $pid)..."
    kill -TERM "$pid" 2>/dev/null || true

    # Wait for process to exit
    local waited=0
    while _pid_alive "$pid" && [[ $waited -lt 10 ]]; do
        sleep 1
        (( waited++ ))
    done

    if _pid_alive "$pid"; then
        echo "WARNING: Daemon did not stop cleanly, sending SIGKILL" >&2
        kill -KILL "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "Daemon stopped"
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_status() {
    local pid
    pid="$(_read_pid)"

    if [[ -z "$pid" ]]; then
        echo "Status: STOPPED (no PID file)"
        return 1
    fi

    if ! _pid_alive "$pid"; then
        echo "Status: STOPPED (stale PID $pid)"
        return 1
    fi

    if _health_check; then
        echo "Status: RUNNING (PID $pid, port ${DAEMON_PORT})"
        return 0
    else
        echo "Status: UNHEALTHY (PID $pid, /health not responding)"
        return 1
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
CMD="${1:-}"
case "$CMD" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    restart) cmd_restart ;;
    status)  cmd_status  ;;
    *)
        echo "Usage: $(basename "$0") start|stop|restart|status" >&2
        exit 1
        ;;
esac
