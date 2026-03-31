#!/usr/bin/env python3
"""session_end.py — Write safety-net memories to SQLite.

Called by session_end.sh AFTER write_log.py has written to markdown.
Takes the filtered extraction JSON (already deduped against markdown log)
and writes any memories not yet in SQLite.

Usage: session_end.py <filtered_extraction_json> <session_id>

No Ollama call here — extraction is done by session_end.sh already.
This only handles the SQLite dedup + write.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("ensemble_memory.session_end")

# ── Locate siblings ────────────────────────────────────────────────────────
_HOOKS_DIR = Path(__file__).parent
sys.path.insert(0, str(_HOOKS_DIR))

import db
import store_memory


def _content_hash(content: str) -> str:
    """SHA-256 of content text for dedup against existing DB memories."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def session_end_sqlite(extraction_json: str, session_id: str) -> tuple[int, int]:
    """Write safety-net memories to SQLite, deduping by content_hash.

    Returns (new_count, already_in_db_count).
    """
    try:
        extraction = json.loads(extraction_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[session_end] Invalid extraction JSON — skipping SQLite write")
        return 0, 0

    memories = extraction.get("memories", [])
    if not memories:
        return 0, 0

    conn = db.get_db()
    new_memories = []
    already_count = 0

    try:
        for mem in memories:
            content = mem.get("content", "").strip()
            if not content:
                continue

            chash = _content_hash(content)
            existing = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?",
                (chash,),
            ).fetchone()

            if existing:
                already_count += 1
            else:
                new_memories.append(mem)
    finally:
        conn.close()

    if not new_memories:
        logger.info(
            "[session_end] SQLite: 0 new, %d already in DB", already_count,
        )
        return 0, already_count

    # Write new memories via store_memory (handles embedding, enrichment, KG, etc.)
    try:
        entities_raw = extraction.get("entities", [])
        new_count, _ = store_memory._store_to_sqlite(
            new_memories, session_id, entities_raw,
        )
    except Exception as exc:
        logger.warning("[session_end] SQLite write failed: %s", exc)
        return 0, already_count

    logger.info(
        "[session_end] SQLite: %d new, %d already in DB",
        new_count, already_count,
    )
    return new_count, already_count


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("Usage: session_end.py <filtered_extraction_json> <session_id>")

    extraction_json = sys.argv[1]
    session_id = sys.argv[2]

    session_end_sqlite(extraction_json, session_id)


if __name__ == "__main__":
    main()
