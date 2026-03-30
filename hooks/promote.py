#!/usr/bin/env python3
"""promote.py — Promote highly-reinforced procedural memories to CLAUDE.md.

Promotion criteria:
  - reinforcement_count >= 5
  - last accessed within 180 days
  - memory_type == 'procedural'

Writes to CLAUDE.md under a '## Learned Behaviors' section.
Uses fcntl advisory locking for concurrent session safety.
"""

import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ensemble_memory.promote")

PROMOTION_THRESHOLD = 5
FRESHNESS_DAYS = 180
LOCK_TIMEOUT = 5  # seconds


def check_and_promote(
    memory_id: str,
    claude_md_path: Optional[str] = None,
) -> bool:
    """Check if a memory qualifies for promotion and write to CLAUDE.md.

    Returns True if promoted, False if skipped or failed.
    """
    import db

    conn = db.get_db()
    try:
        row = conn.execute(
            """
            SELECT content, memory_type, importance, reinforcement_count,
                   last_accessed_at, created_at
            FROM memories WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return False

    # Check criteria
    if row["memory_type"] != "procedural":
        return False
    if row["reinforcement_count"] < PROMOTION_THRESHOLD:
        return False

    # Freshness check
    last_access = row["last_accessed_at"] or row["created_at"]
    days_since = (time.time() - last_access) / 86400
    if days_since > FRESHNESS_DAYS:
        logger.info("[promote] Memory %s too stale (%.0f days)", memory_id[:8], days_since)
        return False

    # Determine target path
    if claude_md_path is None:
        claude_md_path = os.environ.get(
            "ENSEMBLE_MEMORY_CLAUDE_MD",
            str(Path.cwd() / "CLAUDE.md"),
        )

    content = row["content"]
    importance = row["importance"]
    count = row["reinforcement_count"]

    return _write_to_claude_md(
        claude_md_path, content, importance, count
    )


def _write_to_claude_md(
    path: str, content: str, importance: int, count: int,
) -> bool:
    """Append a learned behavior to CLAUDE.md with file locking.

    Idempotent: checks if content already exists before writing.
    Creates the file and section if they don't exist.

    NOTE: All writers to this file must cooperate on fcntl.flock().
    """
    section_header = "## Learned Behaviors"
    entry = f"- {content} (importance: {importance}, reinforced: {count}x)"

    try:
        target = Path(path)

        # Acquire lock FIRST to prevent TOCTOU between read and write.
        # All writers to this file must cooperate on fcntl.flock().
        lock_path = str(target) + ".lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            lock_f = os.fdopen(lock_fd, "w")
            _acquire_lock(lock_f)
        except (OSError, TimeoutError):
            logger.warning("[promote] Could not acquire lock on %s, skipping", path)
            return False

        try:
            # Read existing content (inside lock to prevent TOCTOU)
            existing = ""
            if target.exists():
                try:
                    existing = target.read_text()
                except PermissionError:
                    logger.warning("[promote] CLAUDE.md is read-only: %s", path)
                    return False

            # Idempotency check: skip if content already present
            if content in existing:
                logger.info("[promote] Already promoted: %s", content[:60])
                return False

            # Build new content
            if section_header in existing:
                new_content = existing.replace(
                    section_header,
                    f"{section_header}\n{entry}",
                    1,
                )
            elif existing:
                new_content = existing.rstrip() + f"\n\n{section_header}\n{entry}\n"
            else:
                new_content = f"{section_header}\n{entry}\n"

            target.write_text(new_content)
            logger.info("[promote] Promoted to CLAUDE.md: %s", content[:60])
            return True
        finally:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                lock_f.close()
            except OSError:
                pass

    except Exception as exc:
        logger.warning("[promote] Failed: %s", exc)
        return False


def _acquire_lock(f, timeout: int = LOCK_TIMEOUT) -> None:
    """Acquire an advisory lock with timeout."""
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (BlockingIOError, OSError):
            if time.time() >= deadline:
                raise TimeoutError(f"Could not acquire lock within {timeout}s")
            time.sleep(0.1)
