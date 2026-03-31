#!/usr/bin/env bash
# session_end.sh — Claude Code SessionEnd hook entry point for ensemble memory.
#
# Safety-net scan: runs at session end over the FULL transcript. Catches
# memories that the per-turn Stop hook may have missed (no triage filter here).
# Simple dedup: skip memories whose content already appears in today's log.
#
# Flow:
#   stdin → parse payload → write full transcript to temp file
#     → extract.py (full session, no triage)
#     → dedup against today's log
#     → write_log.py for any new memories

set -euo pipefail

# ── Locate ensemble-memory project root ──────────────────────────────────────
ENSEMBLE_MEMORY_HOME="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="${ENSEMBLE_MEMORY_HOME}/hooks"

# ── Python: use ENSEMBLE_MEMORY_PYTHON if set, else find python3 ──────────────
PYTHON3="${ENSEMBLE_MEMORY_PYTHON:-$(command -v python3)}"

# ── Debug log ──────────────────────────────────────────────────────────────────
DEBUG_LOG="${ENSEMBLE_MEMORY_DIR:-$HOME/.ensemble_memory}/logs/session_end_debug.log"
mkdir -p "$(dirname "$DEBUG_LOG")"

# ── Temp file cleanup ─────────────────────────────────────────────────────────
FULL_TRANSCRIPT_FILE=""
FILTERED_EXTRACTION_FILE=""
cleanup() {
    [[ -n "$FULL_TRANSCRIPT_FILE"    && -f "$FULL_TRANSCRIPT_FILE"    ]] && rm -f "$FULL_TRANSCRIPT_FILE"
    [[ -n "$FILTERED_EXTRACTION_FILE" && -f "$FILTERED_EXTRACTION_FILE" ]] && rm -f "$FILTERED_EXTRACTION_FILE"
}
trap cleanup EXIT

# ── Read hook payload from stdin ─────────────────────────────────────────────
PAYLOAD="$(cat)"

if [[ -z "$PAYLOAD" ]]; then
    exit 0
fi

# ── Extract session_id and transcript_path from payload ──────────────────────
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

# ── Build the full session transcript as a single text file ──────────────────
# All turns, in order, role-prefixed — so extract.py sees complete context.
FULL_TRANSCRIPT_FILE="$(mktemp /tmp/ensemble_memory_session.XXXXXX)"

"$PYTHON3" - "$TRANSCRIPT_PATH" "$FULL_TRANSCRIPT_FILE" <<'PYEOF'
import json
import sys

transcript_path = sys.argv[1]
out_path = sys.argv[2]

turns = []

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

            if msg_type in ("user", "assistant") or role in ("user", "assistant"):
                effective_role = msg_type if msg_type in ("user", "assistant") else role
                content = entry.get("message") or entry.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                content = str(content).strip()
                if content:
                    prefix = "Human" if effective_role == "user" else "Assistant"
                    turns.append(f"{prefix}: {content}")
except OSError:
    pass

# Truncate to last ~4000 chars to stay within model context window
full_text = "\n\n".join(turns)
if len(full_text) > 4000:
    full_text = full_text[-4000:]

with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(full_text + "\n")
PYEOF

# ── Nothing to process if transcript is empty ────────────────────────────────
if [[ ! -s "$FULL_TRANSCRIPT_FILE" ]]; then
    exit 0
fi

# ── Extract memories from the full session (no triage gate) ──────────────────
# Pass empty signals array — we want the LLM to scan everything.
EXTRACTION="$("$PYTHON3" "${HOOKS_DIR}/extract.py" "$FULL_TRANSCRIPT_FILE" "[]" 2>>"$DEBUG_LOG" || echo "")"

if [[ -z "$EXTRACTION" || "$EXTRACTION" == "null" ]]; then
    exit 0
fi

# ── Dedup: check which memories are already in today's log ───────────────────
# Resolve today's log file path (respects ENSEMBLE_MEMORY_LOGS env var).
MEMORY_LOG_DIR="${ENSEMBLE_MEMORY_LOGS:-${HOME}/.ensemble_memory/memory}"
TODAY="$(date +%Y-%m-%d)"
TODAY_LOG="${MEMORY_LOG_DIR}/${TODAY}.md"

# Use python3 to filter out memories whose content already appears in the log.
# grep -F substring check: if content text is found verbatim in the log, skip it.
FILTERED_EXTRACTION_FILE="$(mktemp /tmp/ensemble_memory_filtered.XXXXXX)"

"$PYTHON3" - "$EXTRACTION" "$TODAY_LOG" "$FILTERED_EXTRACTION_FILE" <<'PYEOF'
import json
import sys

extraction_raw = sys.argv[1]
log_path = sys.argv[2]
out_path = sys.argv[3]

try:
    extraction = json.loads(extraction_raw)
except json.JSONDecodeError:
    # Can't parse — write empty result so write_log.py skips gracefully
    with open(out_path, "w") as fh:
        json.dump({"memories": [], "summary": []}, fh)
    sys.exit(0)

# Read today's log if it exists
existing_log = ""
try:
    with open(log_path, encoding="utf-8") as fh:
        existing_log = fh.read()
except OSError:
    pass  # Log doesn't exist yet — all memories are new

memories = extraction.get("memories", [])
new_memories = []

for mem in memories:
    content = mem.get("content", "").strip()
    if not content:
        continue
    # Simple substring dedup: skip if the memory content already appears verbatim
    if content in existing_log:
        continue
    new_memories.append(mem)

result = {
    "memories": new_memories,
    "summary": extraction.get("summary", []) if new_memories else [],
}

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False)
PYEOF

# ── Read the filtered result ──────────────────────────────────────────────────
FILTERED_EXTRACTION="$(cat "$FILTERED_EXTRACTION_FILE")"

# Check if any new memories remain
HAS_MEMORIES="$("$PYTHON3" -c \
    "import json,sys; d=json.loads(sys.argv[1]); print('yes' if d.get('memories') else 'no')" \
    "$FILTERED_EXTRACTION" 2>>"$DEBUG_LOG" || echo "no")"

if [[ "$HAS_MEMORIES" != "yes" ]]; then
    exit 0
fi

# ── Write new memories to the daily log ──────────────────────────────────────
export TRANSCRIPT_PATH
"$PYTHON3" "${HOOKS_DIR}/write_log.py" "$FILTERED_EXTRACTION" "$SESSION_ID" 2>>"$DEBUG_LOG" || true

# ── Write new memories to SQLite (safety-net → recall) ───────────────────
# Dedup by content_hash against existing DB memories. Non-fatal: markdown
# is already written above, SQLite is best-effort.
"$PYTHON3" "${HOOKS_DIR}/session_end.py" "$FILTERED_EXTRACTION" "$SESSION_ID" 2>>"$DEBUG_LOG" || true
