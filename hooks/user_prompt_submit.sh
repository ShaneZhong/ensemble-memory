#!/usr/bin/env bash
# user_prompt_submit.sh — Claude Code UserPromptSubmit hook for ensemble memory.
#
# Triggered AFTER the user submits a prompt, BEFORE Claude processes it.
# Retrieves memories semantically relevant to the prompt and injects them
# as additionalContext so Claude can apply corrections/rules before responding.
#
# Output format (Claude Code hook response):
#   {"additionalContext": "## Relevant Memories...\n..."}
#
# If no relevant memories found (or DB not yet created), outputs:
#   {} (empty JSON — no context injected)
#
# Performance budget: < 500ms total. If the embedding service is slow or
# unavailable, the hook falls back to keyword matching and exits quickly.

set -euo pipefail

# ── Locate ensemble-memory project root ──────────────────────────────────────
ENSEMBLE_MEMORY_HOME="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="${ENSEMBLE_MEMORY_HOME}/hooks"

# ── Python: use ENSEMBLE_MEMORY_PYTHON if set, else find python3 ──────────────
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

# ── Extract session_id and message text from payload ─────────────────────────
# UserPromptSubmit payload: {"session_id": "...", "message": "...", "cwd": "..."}
# The message field may be a plain string or a structured object.
# Prefer jq if available; fall back to python3 -c for portability.
if command -v jq >/dev/null 2>&1; then
    SESSION_ID="$(printf '%s' "$PAYLOAD" | jq -r '.session_id // ""')"
    # Extract message text: handle string, or object with .content (string or array)
    MESSAGE_TEXT="$(printf '%s' "$PAYLOAD" | jq -r '
        if .message | type == "string" then .message
        elif .message | type == "object" then
            if .message.content | type == "string" then .message.content
            elif .message.content | type == "array" then
                [.message.content[] | select(.type == "text") | .text] | join(" ")
            else ""
            end
        else ""
        end
    ')"
    HOOK_CWD="$(printf '%s' "$PAYLOAD" | jq -r '.cwd // ""')"
else
    SESSION_ID="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('session_id', ''))
")"
    MESSAGE_TEXT="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c "
import json, sys
d = json.load(sys.stdin)
msg = d.get('message', '')
if isinstance(msg, str):
    print(msg)
elif isinstance(msg, dict):
    content = msg.get('content', '')
    if isinstance(content, str):
        print(content)
    elif isinstance(content, list):
        parts = [b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text']
        print(' '.join(parts))
    else:
        print('')
else:
    print('')
")"
    HOOK_CWD="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('cwd', ''))
")"
fi

# ── Skip very short prompts (unlikely to match memories meaningfully) ──────────
if [[ ${#MESSAGE_TEXT} -lt 10 ]]; then
    echo "{}"
    exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] session=$SESSION_ID len=${#MESSAGE_TEXT}" >> "$DEBUG_LOG"

# ── Export context vars for the Python script ─────────────────────────────────
export ENSEMBLE_MEMORY_PROJECT="${HOOK_CWD:-$(pwd)}"

# ── Call user_prompt_submit.py with a 450ms timeout guard ─────────────────────
# Run with a timeout so a slow embedding model never blocks Claude.
# GNU timeout / BSD gtimeout / python fallback.
RESULT=""
if command -v timeout >/dev/null 2>&1; then
    RESULT="$(timeout 0.45 "$PYTHON3" "${HOOKS_DIR}/user_prompt_submit.py" \
        "$MESSAGE_TEXT" "$SESSION_ID" 2>>"$DEBUG_LOG" || echo "{}")"
elif command -v gtimeout >/dev/null 2>&1; then
    RESULT="$(gtimeout 0.45 "$PYTHON3" "${HOOKS_DIR}/user_prompt_submit.py" \
        "$MESSAGE_TEXT" "$SESSION_ID" 2>>"$DEBUG_LOG" || echo "{}")"
else
    # No timeout command — run without guard (Python code self-limits where possible)
    RESULT="$("$PYTHON3" "${HOOKS_DIR}/user_prompt_submit.py" \
        "$MESSAGE_TEXT" "$SESSION_ID" 2>>"$DEBUG_LOG" || echo "{}")"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RESULT: ${RESULT:0:120}..." >> "$DEBUG_LOG"
echo "${RESULT:-{\}}"
