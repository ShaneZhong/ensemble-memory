#!/usr/bin/env python3
"""store_memory.py — Write extracted memories to SQLite + daily markdown log.

Usage: store_memory.py <extraction_json> <session_id>

Replaces the direct call to write_log.py in stop.sh.  SQLite errors are
non-fatal: if the DB write fails, markdown is still written (markdown is
the source of truth).

Env vars (passed through to write_log):
    ENSEMBLE_MEMORY_LOGS   Daily .md output dir  (default: ~/.ensemble_memory/memory/)
    ENSEMBLE_MEMORY_DIR    User data root         (default: ~/.ensemble_memory/)
    TRANSCRIPT_PATH        Source transcript path  (metadata only)
"""

import json
import sys
from pathlib import Path

# ── Locate siblings without hardcoded paths ───────────────────────────────────
_HOOKS_DIR = Path(__file__).parent
sys.path.insert(0, str(_HOOKS_DIR))

import os
import db
import write_log

# Promotion threshold: if a procedural memory is reinforced this many times
# within the window, log it as a candidate for CLAUDE.md promotion.
PROMOTION_THRESHOLD = 5
PROJECT = os.environ.get("ENSEMBLE_MEMORY_PROJECT", os.getcwd())


def _store_to_sqlite(memories: list[dict], session_id: str) -> tuple[int, list[str]]:
    """Insert memories into SQLite, detect supersession and reinforcement.

    Returns (new_count, superseded_ids).
    Logs promotion candidates to stderr if threshold is reached.
    """
    new_count = 0
    superseded_ids: list[str] = []

    for mem in memories:
        mem_type = mem.get("type", "")
        trigger = mem.get("trigger_condition", "")

        # ── Reinforcement check (procedural only) ─────────────────────────────
        existing_id = None
        if mem_type == "procedural" and trigger:
            count = db.get_reinforcement_count(trigger)
            if count > 0:
                # Reinforcement exists; skip inserting a duplicate
                if count >= PROMOTION_THRESHOLD:
                    print(
                        f"[store_memory] PROMOTION CANDIDATE: procedural memory "
                        f"'{trigger[:60]}' reinforced {count}x",
                        file=sys.stderr,
                    )
                continue  # don't insert a new row

        # ── Insert new memory ─────────────────────────────────────────────────
        mem_id = db.insert_memory(mem, session_id, PROJECT)
        new_count += 1

        # ── Supersession check (structured: subject+predicate match) ─────────
        subject = mem.get("subject", "")
        predicate = mem.get("predicate", "")
        if subject and predicate:
            superseded = db.detect_supersession(mem_id, subject, predicate)
            if superseded:
                superseded_ids.append(superseded)

        # ── Content-similarity supersession (fallback when no structured triple) ─
        if not subject or not predicate:
            content = mem.get("content", "")
            superseded = db.detect_content_supersession(
                mem_id, content, mem_type, threshold=0.6
            )
            if superseded:
                superseded_ids.append(superseded)

    return new_count, superseded_ids


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("Usage: store_memory.py <extraction_json> <session_id>")

    extraction_raw = sys.argv[1]
    session_id = sys.argv[2]

    # Parse extraction JSON early so we can report a clean error if malformed.
    try:
        extraction = json.loads(extraction_raw)
    except json.JSONDecodeError as exc:
        print(f"[store_memory] invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    memories = extraction.get("memories", [])

    # ── SQLite write (non-fatal) ──────────────────────────────────────────────
    new_count = 0
    superseded_ids: list[str] = []
    db_ok = True
    try:
        new_count, superseded_ids = _store_to_sqlite(memories, session_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[store_memory] SQLite error (continuing): {exc}", file=sys.stderr)
        db_ok = False

    # ── Debug summary ─────────────────────────────────────────────────────────
    if db_ok:
        parts = [f"{new_count} new"]
        if superseded_ids:
            parts.append(f"{len(superseded_ids)} superseded " + " ".join(superseded_ids))
        print(
            f"[store_memory] Stored {len(memories)} memories ({', '.join(parts)})",
            file=sys.stderr,
        )

    # ── Markdown write (always runs) ──────────────────────────────────────────
    # Delegate to write_log.main() by temporarily patching sys.argv so it sees
    # the same arguments it expects.
    _orig_argv = sys.argv[:]
    sys.argv = [str(_HOOKS_DIR / "write_log.py"), extraction_raw, session_id]
    try:
        write_log.main()
    finally:
        sys.argv = _orig_argv


if __name__ == "__main__":
    main()
