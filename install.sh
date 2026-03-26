#!/usr/bin/env bash
# install.sh — Set up the ensemble memory system.
#
# What this does:
#   1. Create ~/.ensemble_memory/{memory,logs/extractions} directories
#   2. Check Ollama is installed
#   3. Pull qwen2.5:3b if not already present
#   4. Copy default_config.toml to ~/.ensemble_memory/config.toml (skip if exists)
#   5. Detect Claude Code hooks.json (project-level first, then user-level)
#   6. Merge Stop + SessionEnd hooks NON-DESTRUCTIVELY into hooks.json
#   7. Make hook scripts executable
#   8. Print success message with test instructions

set -euo pipefail

# ── Locate ensemble-memory project root ──────────────────────────────────────
ENSEMBLE_MEMORY_HOME="$(cd "$(dirname "$0")" && pwd)"
HOOKS_DIR="${ENSEMBLE_MEMORY_HOME}/hooks"
CONFIG_DIR="${ENSEMBLE_MEMORY_HOME}/config"

echo "==> ensemble-memory install"
echo "    Project root: ${ENSEMBLE_MEMORY_HOME}"
echo ""

# ── 1. Create user data directories ──────────────────────────────────────────
echo "--> Creating ~/.ensemble_memory directories..."
mkdir -p \
    "${HOME}/.ensemble_memory/memory" \
    "${HOME}/.ensemble_memory/logs/extractions"
echo "    OK: ~/.ensemble_memory/{memory,logs/extractions}"

# ── 2. Check Ollama is installed ─────────────────────────────────────────────
echo "--> Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    echo ""
    echo "ERROR: Ollama is not installed."
    echo "  macOS:  brew install ollama"
    echo "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi
echo "    OK: ollama found at $(command -v ollama)"

# ── 3. Pull qwen2.5:3b if not already present ────────────────────────────────
echo "--> Checking for qwen2.5:3b model..."
if ollama list 2>/dev/null | grep -q "qwen2.5:3b"; then
    echo "    OK: qwen2.5:3b already present, skipping pull"
else
    echo "    Pulling qwen2.5:3b (this may take a few minutes)..."
    ollama pull qwen2.5:3b
    echo "    OK: qwen2.5:3b pulled"
fi

# ── 4. Copy default config ────────────────────────────────────────────────────
DEFAULT_CONFIG="${CONFIG_DIR}/default_config.toml"
USER_CONFIG="${HOME}/.ensemble_memory/config.toml"

echo "--> Checking config..."
if [[ -f "$USER_CONFIG" ]]; then
    echo "    SKIP: ~/.ensemble_memory/config.toml already exists (not overwriting)"
elif [[ -f "$DEFAULT_CONFIG" ]]; then
    cp "$DEFAULT_CONFIG" "$USER_CONFIG"
    echo "    OK: copied default_config.toml -> ~/.ensemble_memory/config.toml"
else
    echo "    SKIP: no default_config.toml found at ${DEFAULT_CONFIG} (config not copied)"
fi

# ── 5. Detect Claude Code hooks.json location ────────────────────────────────
echo "--> Locating Claude Code hooks.json..."

# Check project-level first (.claude/hooks.json relative to CWD)
PROJECT_HOOKS="${PWD}/.claude/hooks.json"
USER_HOOKS="${HOME}/.claude/hooks.json"

if [[ -f "$PROJECT_HOOKS" ]]; then
    HOOKS_JSON="$PROJECT_HOOKS"
    echo "    Found project-level hooks: ${HOOKS_JSON}"
elif [[ -f "$USER_HOOKS" ]]; then
    HOOKS_JSON="$USER_HOOKS"
    echo "    Found user-level hooks: ${HOOKS_JSON}"
else
    # Neither exists — create user-level hooks.json
    HOOKS_JSON="$USER_HOOKS"
    mkdir -p "$(dirname "$HOOKS_JSON")"
    echo '{}' > "$HOOKS_JSON"
    echo "    Created: ${HOOKS_JSON}"
fi

# ── 6. Merge Stop + SessionEnd hooks non-destructively ───────────────────────
echo "--> Merging ensemble-memory hooks into ${HOOKS_JSON}..."

# Use python3 to parse and merge — jq may not be available on all systems.
python3 - "${HOOKS_JSON}" "${HOOKS_DIR}/stop.sh" "${HOOKS_DIR}/session_end.sh" <<'PYEOF'
import json
import sys

hooks_path   = sys.argv[1]
stop_script  = sys.argv[2]
session_script = sys.argv[3]

# Load existing hooks.json
try:
    with open(hooks_path, encoding="utf-8") as fh:
        config = json.load(fh)
except (OSError, json.JSONDecodeError):
    config = {}

if not isinstance(config, dict):
    config = {}

# Ensure top-level "hooks" key exists
if "hooks" not in config or not isinstance(config["hooks"], dict):
    config["hooks"] = {}

hooks = config["hooks"]

# Helper: merge a single hook entry without duplicating
def merge_hook(event_name: str, command: str, timeout_ms: int) -> None:
    existing = hooks.get(event_name, [])
    if not isinstance(existing, list):
        existing = []

    # Check if this command is already registered
    for entry in existing:
        if isinstance(entry, dict) and entry.get("command") == command:
            # Already present — update timeout in case it changed
            entry["timeout"] = timeout_ms
            hooks[event_name] = existing
            return

    # Not present — append
    existing.append({"command": command, "timeout": timeout_ms})
    hooks[event_name] = existing

merge_hook("Stop",       stop_script,    60000)   # 60s — covers cold model load + extraction (7-9s warm)
merge_hook("SessionEnd", session_script, 120000)  # 120s — full session scan can be longer

config["hooks"] = hooks

with open(hooks_path, "w", encoding="utf-8") as fh:
    json.dump(config, fh, indent=2, ensure_ascii=False)
    fh.write("\n")

print(f"    OK: hooks merged into {hooks_path}")
PYEOF

# ── 7. Make hook scripts executable ──────────────────────────────────────────
echo "--> Making hook scripts executable..."
chmod +x "${HOOKS_DIR}/stop.sh"
chmod +x "${HOOKS_DIR}/session_end.sh"
echo "    OK: stop.sh, session_end.sh are executable"

# ── 8. Success message ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ensemble-memory installed successfully!"
echo "============================================================"
echo ""
echo "Hooks registered in: ${HOOKS_JSON}"
echo "Memory log dir:       ~/.ensemble_memory/memory/"
echo "Project root:         ${ENSEMBLE_MEMORY_HOME}"
echo ""
echo "To test:"
echo "  1. Start a Claude Code session"
echo "  2. Say: \"no, don't use MySQL, use PostgreSQL for this project\""
echo "  3. End the session"
echo "  4. Check: cat ~/.ensemble_memory/memory/\$(date +%Y-%m-%d).md"
echo ""
echo "Environment variables you can set to override defaults:"
echo "  ENSEMBLE_MEMORY_HOME   Project root (auto-detected)"
echo "  ENSEMBLE_MEMORY_DIR    User data root (default: ~/.ensemble_memory)"
echo "  ENSEMBLE_MEMORY_LOGS   Memory log dir (default: ~/.ensemble_memory/memory)"
echo "  OLLAMA_HOST            Ollama API (default: http://localhost:11434)"
echo "  ENSEMBLE_MEMORY_MODEL  Model name (default: qwen2.5:3b)"
