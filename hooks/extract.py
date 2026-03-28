#!/usr/bin/env python3
"""Ollama extraction component for ensemble memory system.

Usage: extract.py <turn_file> <signal_hints_json>
  turn_file         Path to temp file containing the conversation turn text
  signal_hints_json JSON string of signal hints from triage.py

Outputs extraction JSON to stdout on success.
Writes to missed_turns.jsonl and exits 1 on unrecoverable failure.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Config from environment ───────────────────────────────────────────────────

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("ENSEMBLE_MEMORY_MODEL", "qwen2.5:3b")
ENSEMBLE_MEMORY_DIR = Path(
    os.environ.get("ENSEMBLE_MEMORY_DIR", Path.home() / ".ensemble_memory")
)
PROMPTS_DIR = Path(
    os.environ.get(
        "ENSEMBLE_MEMORY_HOME",
        Path(__file__).parent.parent,
    )
) / "hooks" / "prompts"

MISSED_TURNS_FILE = ENSEMBLE_MEMORY_DIR / "logs" / "missed_turns.jsonl"
STATS_FILE = ENSEMBLE_MEMORY_DIR / "logs" / "extraction_stats.jsonl"

TIMEOUT_SECONDS = int(os.environ.get("ENSEMBLE_MEMORY_TIMEOUT", "30"))  # 30s default; full extraction takes 7-9s on M4 + cold load can add 2-5s
TEMPERATURE = 0.2
NUM_PREDICT = 1024

# ── Helpers ───────────────────────────────────────────────────────────────────

def log_missed_turn(turn_text: str, reason: str) -> None:
    MISSED_TURNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "turn_text": turn_text,
    }
    with open(MISSED_TURNS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_stats(success: bool, latency_ms: int, retry: bool) -> None:
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "latency_ms": latency_ms,
        "retry": retry,
    }
    with open(STATS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def call_ollama(prompt: str) -> dict:
    """Call Ollama /api/generate with constrained JSON output.

    Returns parsed response dict.
    Raises urllib.error.URLError on network/timeout error.
    Raises json.JSONDecodeError if response body is not valid JSON.
    """
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        body = resp.read().decode("utf-8")

    return json.loads(body)


def validate_extraction(data: dict) -> bool:
    """Return True if data has valid extraction structure."""
    if not isinstance(data.get("memories"), list):
        return False
    if not isinstance(data.get("summary"), list):
        return False
    # entities and relationships are optional but must be lists if present
    if "entities" in data and not isinstance(data["entities"], list):
        return False
    if "relationships" in data and not isinstance(data["relationships"], list):
        return False
    return True


def load_prompt_template() -> str:
    prompt_path = PROMPTS_DIR / "extraction.txt"
    return prompt_path.read_text(encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <turn_file> <signal_hints_json>", file=sys.stderr)
        return 1

    turn_file = Path(sys.argv[1])
    signal_hints_raw = sys.argv[2]

    turn_text = turn_file.read_text(encoding="utf-8")

    # Smart truncation: keep user text fully (contains corrections/decisions),
    # truncate assistant text (usually long code/explanations).
    # Turn format from stop.sh: "Human: ...\n\nAssistant: ..."
    MAX_TURN_CHARS = 6000
    if len(turn_text) > MAX_TURN_CHARS:
        # Split at "Assistant:" boundary to preserve user text
        split_marker = "\n\nAssistant:"
        split_idx = turn_text.find(split_marker)
        if split_idx > 0:
            user_part = turn_text[:split_idx]
            assistant_part = turn_text[split_idx + len(split_marker):]
            remaining = MAX_TURN_CHARS - len(user_part) - len(split_marker)
            if remaining > 200:
                turn_text = user_part + split_marker + assistant_part[:remaining] + "\n[...truncated...]"
            else:
                # User text alone exceeds limit — truncate it too
                turn_text = turn_text[:MAX_TURN_CHARS] + "\n[...truncated...]"
        else:
            turn_text = turn_text[:MAX_TURN_CHARS] + "\n[...truncated...]"

    template = load_prompt_template()
    prompt = template.replace("{signal_hints}", signal_hints_raw).replace(
        "{conversation_turn}", turn_text
    )

    start_ms = time.monotonic()
    retry = False
    extraction = None

    # First attempt
    try:
        response = call_ollama(prompt)
        raw = json.loads(response.get("response", "{}"))
        if validate_extraction(raw):
            extraction = raw
    except (urllib.error.URLError, TimeoutError) as exc:
        latency_ms = int((time.monotonic() - start_ms) * 1000)
        log_missed_turn(turn_text, f"timeout: {exc}")
        log_stats(success=False, latency_ms=latency_ms, retry=False)
        return 1
    except (json.JSONDecodeError, KeyError):
        pass  # fall through to retry

    # Retry once with simplified summary-only prompt
    if extraction is None:
        retry = True
        simplified_prompt = (
            "You are a memory extraction system. "
            "Read the following conversation turn and return ONLY valid JSON "
            'with this exact schema: {"memories": [], "summary": ["<one sentence summary>"]}\n\n'
            f"Conversation turn:\n{turn_text}"
        )
        try:
            response = call_ollama(simplified_prompt)
            raw = json.loads(response.get("response", "{}"))
            if validate_extraction(raw):
                extraction = raw
        except (urllib.error.URLError, TimeoutError) as exc:
            latency_ms = int((time.monotonic() - start_ms) * 1000)
            log_missed_turn(turn_text, f"timeout on retry: {exc}")
            log_stats(success=False, latency_ms=latency_ms, retry=True)
            return 1
        except (json.JSONDecodeError, KeyError):
            pass

    latency_ms = int((time.monotonic() - start_ms) * 1000)

    if extraction is None:
        log_missed_turn(turn_text, "schema validation failed after retry")
        log_stats(success=False, latency_ms=latency_ms, retry=retry)
        return 1

    log_stats(success=True, latency_ms=latency_ms, retry=retry)
    print(json.dumps(extraction, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
