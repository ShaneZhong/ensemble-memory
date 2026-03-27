#!/usr/bin/env bash
# user_prompt_submit.sh — Claude Code UserPromptSubmit hook for ensemble memory.
#
# Triggered AFTER the user submits a prompt, BEFORE Claude processes it.
# Delegates to user_prompt_submit.py which calls the embedding daemon.
# Target: < 50ms (HTTP to localhost, daemon keeps model warm).
#
# Output: {"additionalContext": "..."} or {}

set -euo pipefail

# ── Locate ensemble-memory project root ──────────────────────────────────────
ENSEMBLE_MEMORY_HOME="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="${ENSEMBLE_MEMORY_HOME}/hooks"

# ── Python: use ENSEMBLE_MEMORY_PYTHON if set, else python3 ──────────────────
PYTHON3="${ENSEMBLE_MEMORY_PYTHON:-$(command -v python3)}"

# ── Debug log ─────────────────────────────────────────────────────────────────
DEBUG_LOG="${ENSEMBLE_MEMORY_DIR:-$HOME/.ensemble_memory}/logs/user_prompt_debug.log"
mkdir -p "$(dirname "$DEBUG_LOG")"

# ── Read hook payload from stdin ─────────────────────────────────────────────
PAYLOAD="$(cat)"

if [[ -z "$PAYLOAD" ]]; then
    echo "{}"
    exit 0
fi

# ── Extract session_id, message text, and cwd from payload ───────────────────
# Prefer jq if available; fall back to python3 for portability.
# Claude Code sends: {"session_id":"...", "prompt":"user text", "cwd":"..."}
if command -v jq >/dev/null 2>&1; then
    SESSION_ID="$(printf '%s' "$PAYLOAD" | jq -r '.session_id // ""')"
    MESSAGE_TEXT="$(printf '%s' "$PAYLOAD" | jq -r '.prompt // ""')"
    HOOK_CWD="$(printf '%s' "$PAYLOAD" | jq -r '.cwd // ""')"
else
    SESSION_ID="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c "
import json, sys; d=json.load(sys.stdin); print(d.get('session_id',''))")"
    MESSAGE_TEXT="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c "
import json, sys; d=json.load(sys.stdin); print(d.get('prompt',''))")"
    HOOK_CWD="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c "
import json, sys; d=json.load(sys.stdin); print(d.get('cwd',''))")"
fi

# ── Skip very short prompts ───────────────────────────────────────────────────
if [[ ${#MESSAGE_TEXT} -lt 10 ]]; then
    echo "{}"
    exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session=$SESSION_ID len=${#MESSAGE_TEXT}" >> "$DEBUG_LOG"

# ── Export context for Python script ─────────────────────────────────────────
export ENSEMBLE_MEMORY_PROJECT="${HOOK_CWD:-$(pwd)}"

# ── Call user_prompt_submit.py (thin HTTP client — no timeout needed) ─────────
RESULT="$("$PYTHON3" "${HOOKS_DIR}/user_prompt_submit.py" \
    "$MESSAGE_TEXT" "$SESSION_ID" 2>>"$DEBUG_LOG" || echo "{}")"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RESULT: ${RESULT:0:120}" >> "$DEBUG_LOG"
echo "${RESULT:-{}}"
