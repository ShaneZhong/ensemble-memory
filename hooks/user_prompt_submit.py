#!/usr/bin/env python3
"""user_prompt_submit.py — Thin HTTP client to the ensemble-memory embedding daemon.

Usage: user_prompt_submit.py <message_text> [session_id]

Env vars:
    ENSEMBLE_MEMORY_PROJECT      Project path for scoping (default: cwd)
    ENSEMBLE_MEMORY_DAEMON_PORT  Daemon port (default: 9876)

Output:
    {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "..."}} — when relevant
    {}  — when nothing found or daemon down
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DAEMON_PORT = int(os.environ.get("ENSEMBLE_MEMORY_DAEMON_PORT", "9876"))
DAEMON_BASE = f"http://127.0.0.1:{DAEMON_PORT}"

# ── Locate daemon_ctl.sh for auto-start ───────────────────────────────────────
_HOOKS_DIR  = Path(__file__).resolve().parent
_DAEMON_CTL = _HOOKS_DIR.parent / "daemon" / "daemon_ctl.sh"


def _daemon_running() -> bool:
    """Quick /health check — returns True if daemon responds within 200ms."""
    try:
        urllib.request.urlopen(f"{DAEMON_BASE}/health", timeout=0.2)
        return True
    except Exception:
        return False


def _try_start_daemon() -> None:
    """Fire-and-forget daemon start. Non-blocking — daemon won't be ready yet."""
    import subprocess
    if not _DAEMON_CTL.exists():
        return
    try:
        subprocess.Popen(
            ["bash", str(_DAEMON_CTL), "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception:
        pass


def _search(query: str, project: str) -> dict:
    """POST /search to daemon. Returns parsed JSON or empty dict on any error."""
    payload = json.dumps({"query": query, "project": project}).encode()
    req = urllib.request.Request(
        f"{DAEMON_BASE}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.4) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def main() -> None:
    # ── Parse args ────────────────────────────────────────────────────────────
    message = sys.argv[1] if len(sys.argv) >= 2 else ""
    if not message:
        message = sys.stdin.read().strip()

    project = os.environ.get("ENSEMBLE_MEMORY_PROJECT", "") or os.getcwd()

    if not message or len(message) < 10:
        print("{}")
        return

    # ── Check daemon health; auto-start if down ────────────────────────────────
    if not _daemon_running():
        _try_start_daemon()
        # Daemon is warming up — return {} for this prompt; next prompt will hit it
        print("{}")
        return

    # ── Query daemon ──────────────────────────────────────────────────────────
    result = _search(message, project)
    context = result.get("context", "")

    if not context:
        print("{}")
        return

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
