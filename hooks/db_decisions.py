#!/usr/bin/env python3
"""db_decisions.py — Decision vault: insert, search, and FTS5 helpers.

Split from db.py for maintainability. All functions import get_db from db.
"""

import json
import logging
import re
import time
import uuid
from typing import Optional

logger = logging.getLogger("ensemble_memory.db")


_VALID_DECISION_TYPES = frozenset([
    "ARCHITECTURAL", "PREFERENCE", "ERROR_RESOLUTION", "CONSTRAINT", "PATTERN"
])


def insert_decision(
    memory_id: str,
    decision_type: str,
    content_hash: str,
    keywords: Optional[list[str]] = None,
    files_referenced: Optional[list[str]] = None,
    project: str = "",
    session_id: str = "",
) -> Optional[str]:
    """Insert a decision record linked to a memory. Returns decision id.

    Skips if content_hash already exists (exact dedup). Normalizes
    decision_type with fallback to None (skip insertion).
    """
    # Normalize decision_type
    if decision_type not in _VALID_DECISION_TYPES:
        for part in (decision_type or "").replace(",", "|").split("|"):
            candidate = part.strip().upper()
            if candidate in _VALID_DECISION_TYPES:
                decision_type = candidate
                break
        else:
            return None  # Not a valid decision -- skip

    from db import get_db
    conn = get_db()
    now = time.time()

    # Exact dedup
    existing = conn.execute(
        "SELECT id FROM decisions WHERE content_hash = ? AND project = ?",
        (content_hash, project),
    ).fetchone()
    if existing:
        conn.close()
        return existing["id"]

    decision_id = str(uuid.uuid4())
    try:
        conn.execute(
            """
            INSERT INTO decisions (
                id, memory_id, decision_type, content_hash,
                keywords, files_referenced, project, session_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id, memory_id, decision_type, content_hash,
                json.dumps(keywords or []),
                json.dumps(files_referenced or []),
                project, session_id, now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return decision_id


def _sanitize_fts5_query(query: str) -> str:
    """Escape FTS5 special characters by extracting alphanumeric tokens and quoting them."""
    tokens = re.findall(r'[a-zA-Z0-9_]+', query)
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


# NOTE: Daemon _bm25_search has a similar FTS5 query. Keep them in sync.
def search_decisions_bm25(query: str, project: str = "", limit: int = 10) -> list[dict]:
    """Search decisions using FTS5 BM25 ranking. Returns list of dicts."""
    from db import get_db
    conn = get_db()
    try:
        project_clause = ""
        params: list = [_sanitize_fts5_query(query)]
        if project:
            project_clause = "AND d.project = ?"
            params.append(project)
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT d.id, d.memory_id, d.decision_type, d.keywords,
                   d.files_referenced, d.project, d.created_at,
                   m.content, m.importance, m.memory_type,
                   bm25(decisions_fts) AS bm25_score
            FROM decisions_fts
            JOIN decisions d ON d.rowid = decisions_fts.rowid
            JOIN memories m ON m.id = d.memory_id
            WHERE decisions_fts MATCH ?
              {project_clause}
            ORDER BY bm25(decisions_fts)
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [dict(row) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()
