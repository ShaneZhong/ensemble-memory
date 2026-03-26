#!/usr/bin/env bash
# session_start.sh — Claude Code SessionStart hook entry point for ensemble memory.
#
# Triggered at the start of each Claude Code session. Reads hook payload from
# stdin, calls session_start.py to load high-importance memories from the DB,
# and passes the JSON context response through to stdout for Claude Code to
# inject as additionalContext.
#
# Output format (Claude Code hook response):
#   {"additionalContext": "## Ensemble Memory: Standing Context\n..."}
#
# If no memories exist (first run, empty DB, or DB not yet created), outputs:
#   {} (empty JSON — no context injected)

set -euo pipefail

# ── Locate ensemble-memory project root ──────────────────────────────────────
ENSEMBLE_MEMORY_HOME="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="${ENSEMBLE_MEMORY_HOME}/hooks"

# ── Python: use ENSEMBLE_MEMORY_PYTHON if set, else find python3 ──────────────
PYTHON3="${ENSEMBLE_MEMORY_PYTHON:-$(command -v python3)}"

# ── Debug log ──────────────────────────────────────────────────────────────────
DEBUG_LOG="${ENSEMBLE_MEMORY_DIR:-$HOME/.ensemble_memory}/logs/session_start_debug.log"
mkdir -p "$(dirname "$DEBUG_LOG")"

# ── Read hook payload from stdin ─────────────────────────────────────────────
PAYLOAD="$(cat)"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PAYLOAD: ${PAYLOAD:-EMPTY}" >> "$DEBUG_LOG"

if [[ -z "$PAYLOAD" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] EXIT: empty payload" >> "$DEBUG_LOG"
    echo "{}"
    exit 0
fi

# ── Extract session_id from payload ──────────────────────────────────────────
# Prefer jq if available; fall back to python3 -c for portability.
if command -v jq >/dev/null 2>&1; then
    SESSION_ID="$(printf '%s' "$PAYLOAD" | jq -r '.session_id // ""')"
else
    SESSION_ID="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c \
        "import json,sys; d=json.load(sys.stdin); print(d.get('session_id',''))")"
fi

# ── Call session_start.py and pass its JSON output through ───────────────────
# session_start.py outputs {"additionalContext": "..."} or {} on no memories.
# Extract cwd from payload for project scoping
if command -v jq >/dev/null 2>&1; then
    HOOK_CWD="$(printf '%s' "$PAYLOAD" | jq -r '.cwd // ""')"
else
    HOOK_CWD="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c \
        "import json,sys; d=json.load(sys.stdin); print(d.get('cwd',''))")"
fi
export ENSEMBLE_MEMORY_PROJECT="${HOOK_CWD:-$(pwd)}"

RESULT="$("$PYTHON3" "${HOOKS_DIR}/session_start.py" "$SESSION_ID" 2>>"$DEBUG_LOG" || echo "{}")"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RESULT: ${RESULT:-EMPTY}" >> "$DEBUG_LOG"
echo "$RESULT"
