#!/usr/bin/env bash
# stop.sh — Claude Code Stop hook entry point for ensemble memory system.
#
# Triggered after every agent response. Reads the hook payload from stdin,
# extracts the latest conversation turn, runs triage, and (if signals found)
# runs extraction + writes to the daily memory log.
#
# Flow:
#   stdin → parse payload → extract latest turn → triage.py
#     → signals?  no  → exit 0 (fast path, < 10ms total)
#               yes  → extract.py → store_memory.py (SQLite + markdown)

set -euo pipefail

# ── Locate ensemble-memory project root ──────────────────────────────────────
ENSEMBLE_MEMORY_HOME="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="${ENSEMBLE_MEMORY_HOME}/hooks"

# ── Python: use ENSEMBLE_MEMORY_PYTHON if set, else find python3 ──────────────
PYTHON3="${ENSEMBLE_MEMORY_PYTHON:-$(command -v python3)}"

# ── Temp file cleanup ─────────────────────────────────────────────────────────
TURN_FILE=""
cleanup() {
    [[ -n "$TURN_FILE" && -f "$TURN_FILE" ]] && rm -f "$TURN_FILE"
    [[ -n "$USER_ONLY_FILE" && -f "$USER_ONLY_FILE" ]] && rm -f "$USER_ONLY_FILE"
}
USER_ONLY_FILE=""
trap cleanup EXIT

# ── Read hook payload from stdin ─────────────────────────────────────────────
PAYLOAD="$(cat)"

if [[ -z "$PAYLOAD" ]]; then
    exit 0
fi

# ── Check stop_hook_active to prevent infinite loops ─────────────────────────
# Per Claude Code docs: stop_hook_active=true means Claude is already continuing
# from a previous Stop hook. Skip processing to avoid re-triggering.
if command -v jq >/dev/null 2>&1; then
    STOP_ACTIVE="$(printf '%s' "$PAYLOAD" | jq -r '.stop_hook_active // false')"
else
    STOP_ACTIVE="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c \
        "import json,sys; d=json.load(sys.stdin); print(str(d.get('stop_hook_active',False)).lower())")"
fi
if [[ "$STOP_ACTIVE" == "true" ]]; then
    exit 0
fi

# ── Extract session_id and transcript_path from payload ──────────────────────
# Claude Code Stop hook sends: {session_id, transcript_path, cwd, permission_mode,
#   hook_event_name, stop_hook_active, last_assistant_message}
if command -v jq >/dev/null 2>&1; then
    SESSION_ID="$(printf '%s' "$PAYLOAD" | jq -r '.session_id // ""')"
    TRANSCRIPT_PATH="$(printf '%s' "$PAYLOAD" | jq -r '.transcript_path // ""')"
else
    SESSION_ID="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c \
        "import json,sys; d=json.load(sys.stdin); print(d.get('session_id',''))")"
    TRANSCRIPT_PATH="$(printf '%s' "$PAYLOAD" | "$PYTHON3" -c \
        "import json,sys; d=json.load(sys.stdin); print(d.get('transcript_path',''))")"
fi

# ── Validate we have a usable transcript ─────────────────────────────────────
if [[ -z "$TRANSCRIPT_PATH" || ! -f "$TRANSCRIPT_PATH" ]]; then
    exit 0
fi

# ── Extract the latest conversation turn from the JSONL transcript ───────────
# Parse JSONL, find last "user" message and last "assistant" message, combine.
TURN_FILE="$(mktemp /tmp/ensemble_memory_turn.XXXXXX)"

"$PYTHON3" - "$TRANSCRIPT_PATH" "$TURN_FILE" <<'PYEOF'
import json
import sys

transcript_path = sys.argv[1]
turn_file = sys.argv[2]

last_user = ""
last_assistant = ""

try:
    with open(transcript_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = entry.get("type", "")
            role = entry.get("role", "")

            # Handle flat {type, message} and nested {role, content} formats.
            # message can be: str, dict (with 'content' key), or list of blocks.
            def extract_text(raw):
                """Extract plain text from any message format."""
                if isinstance(raw, str):
                    return raw.strip()
                if isinstance(raw, dict):
                    inner = raw.get("content", "")
                    return extract_text(inner)  # recurse
                if isinstance(raw, list):
                    texts = []
                    for block in raw:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                texts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                pass  # skip tool results — not user text
                    return " ".join(texts).strip()
                return ""

            if msg_type == "user" or role == "user":
                content = extract_text(entry.get("message") or entry.get("content") or "")
                if content:
                    last_user = content
            elif msg_type == "assistant" or role == "assistant":
                content = extract_text(entry.get("message") or entry.get("content") or "")
                if content:
                    last_assistant = content
except OSError:
    pass

turn_text = ""
if last_user:
    turn_text += f"Human: {last_user}\n\n"
if last_assistant:
    turn_text += f"Assistant: {last_assistant}\n"

with open(turn_file, "w", encoding="utf-8") as fh:
    fh.write(turn_text)
PYEOF

# ── Nothing to triage if turn file is empty ──────────────────────────────────
if [[ ! -s "$TURN_FILE" ]]; then
    exit 0
fi

# ── Extract user-only text for triage (avoid false positives from assistant) ──
USER_ONLY_FILE="$(mktemp /tmp/ensemble_memory_user.XXXXXX)"
"$PYTHON3" -c "
import sys
text = open(sys.argv[1]).read()
lines = text.splitlines()
user_lines = []
in_user = False
for line in lines:
    if line.startswith(('Human:', 'User:')):
        in_user = True
        user_lines.append(line)
    elif line.startswith(('Assistant:', 'Claude:')):
        in_user = False
    elif in_user:
        user_lines.append(line)
with open(sys.argv[2], 'w') as f:
    f.write('\n'.join(user_lines))
" "$TURN_FILE" "$USER_ONLY_FILE"

# ── Triage: check for correction/decision signals (user text only) ───────────
SIGNALS="$("$PYTHON3" "${HOOKS_DIR}/triage.py" "$USER_ONLY_FILE" 2>/dev/null || echo "[]")"

# Fast path: no signals found
if [[ "$SIGNALS" == "[]" || -z "$SIGNALS" ]]; then
    exit 0
fi

# ── Extract memories via Ollama ───────────────────────────────────────────────
EXTRACTION="$("$PYTHON3" "${HOOKS_DIR}/extract.py" "$TURN_FILE" "$SIGNALS" 2>/dev/null || echo "")"

if [[ -z "$EXTRACTION" || "$EXTRACTION" == "null" ]]; then
    exit 0
fi

# ── Write to SQLite + daily memory log ───────────────────────────────────────
export TRANSCRIPT_PATH
"$PYTHON3" "${HOOKS_DIR}/store_memory.py" "$EXTRACTION" "$SESSION_ID" 2>/dev/null || true
