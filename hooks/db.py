#!/usr/bin/env python3
"""db.py — Shared SQLite database module for the ensemble memory system.

All hooks import this module to access the temporal hub (memories table)
and supporting tables. Handles DB initialization, WAL mode, and the core
temporal scoring logic.

Env vars:
    ENSEMBLE_MEMORY_DIR   Root data directory (default: ~/.ensemble_memory)
"""

import hashlib
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

# Test override: set _DB_PATH_OVERRIDE to redirect DB to a temp path
_DB_PATH_OVERRIDE: Optional[str] = None


def _db_path() -> Path:
    if _DB_PATH_OVERRIDE:
        p = Path(_DB_PATH_OVERRIDE)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    root = Path(
        os.environ.get("ENSEMBLE_MEMORY_DIR", Path.home() / ".ensemble_memory")
    ).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / "memory.db"


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memories (
    id                    TEXT PRIMARY KEY,
    content               TEXT NOT NULL,
    content_hash          TEXT NOT NULL,
    memory_type           TEXT NOT NULL
                          CHECK(memory_type IN ('episodic','semantic','procedural','correction')),
    importance            INTEGER NOT NULL DEFAULT 5
                          CHECK(importance BETWEEN 1 AND 10),
    extraction_confidence REAL NOT NULL DEFAULT 0.8
                          CHECK(extraction_confidence BETWEEN 0.0 AND 1.0),
    confidence            REAL NOT NULL DEFAULT 1.0
                          CHECK(confidence BETWEEN 0.0 AND 1.0),
    subject               TEXT,
    predicate             TEXT,
    object                TEXT,
    session_id            TEXT,
    source_expert         TEXT,
    project               TEXT,
    created_at            REAL NOT NULL,
    event_time            REAL,
    valid_from            REAL,
    valid_to              REAL,
    last_accessed_at      REAL,
    access_count          INTEGER NOT NULL DEFAULT 0,
    superseded_by         TEXT REFERENCES memories(id),
    superseded_at         REAL,
    decay_rate            REAL NOT NULL DEFAULT 0.16,
    stability             REAL NOT NULL DEFAULT 0.0,
    temporal_confidence   REAL NOT NULL DEFAULT 1.0,
    reinforcement_count   INTEGER NOT NULL DEFAULT 0,
    promotion_candidate   INTEGER NOT NULL DEFAULT 0,
    temporal_score        REAL,
    score_computed_at     REAL,
    gc_eligible           INTEGER NOT NULL DEFAULT 0,
    gc_protected          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_valid_window  ON memories(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_memories_superseded    ON memories(superseded_by)
    WHERE superseded_by IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_created       ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_type          ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance    ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_content_hash  ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_gc            ON memories(gc_eligible)
    WHERE gc_eligible = 1;
CREATE INDEX IF NOT EXISTS idx_memories_session       ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_subject       ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_memories_validity      ON memories(valid_to, superseded_by, gc_eligible);
CREATE INDEX IF NOT EXISTS idx_memories_project       ON memories(project);

CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,
    started_at       REAL NOT NULL,
    ended_at         REAL,
    duration_seconds REAL,
    summary          TEXT
);

CREATE TABLE IF NOT EXISTS supersession_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    old_memory_id           TEXT NOT NULL REFERENCES memories(id),
    new_memory_id           TEXT NOT NULL REFERENCES memories(id),
    event_time              REAL NOT NULL,
    detected_by             TEXT NOT NULL,
    processed_by_temporal   INTEGER DEFAULT 0,
    processed_by_kg         INTEGER DEFAULT 0,
    processed_by_contextual INTEGER DEFAULT 0,
    created_at              REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_supersession_unprocessed ON supersession_events(
    processed_by_temporal, processed_by_kg, processed_by_contextual);

CREATE TABLE IF NOT EXISTS semantic_chunk_map (
    chunk_hash        TEXT PRIMARY KEY,
    memory_id         TEXT NOT NULL REFERENCES memories(id),
    milvus_collection TEXT DEFAULT 'memory_chunks'
);
CREATE INDEX IF NOT EXISTS idx_chunk_map_memory ON semantic_chunk_map(memory_id);

CREATE TABLE IF NOT EXISTS lambda_base_constants (
    memory_type            TEXT PRIMARY KEY
                           CHECK(memory_type IN ('procedural','correction','semantic','episodic')),
    lambda_base            REAL NOT NULL,
    nominal_half_life_days REAL NOT NULL,
    description            TEXT
);

INSERT OR IGNORE INTO lambda_base_constants VALUES
    ('procedural', 0.01, 69.3, 'Slow decay -- procedures remain relevant'),
    ('correction', 0.05, 13.9, 'Moderate decay -- corrections age out'),
    ('semantic',   0.10,  6.9, 'Standard decay -- facts need refreshing'),
    ('episodic',   0.16,  4.3, 'Fast decay -- narratives lose relevance');

CREATE TABLE IF NOT EXISTS supersession_depth_limits (
    memory_type     TEXT PRIMARY KEY
                    CHECK(memory_type IN ('procedural','correction','semantic','episodic')),
    max_chain_depth INTEGER NOT NULL,
    description     TEXT
);

INSERT OR IGNORE INTO supersession_depth_limits VALUES
    ('procedural', 3, 'Architectural/constraint chains pruned at depth 3'),
    ('correction', 2, 'Error resolution chains pruned aggressively'),
    ('semantic',   3, 'Fact chains pruned at depth 3'),
    ('episodic',   5, 'Preference chains allowed longer history');
"""

# ── Connection ─────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection, creating tables on first run."""
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ── Internal helpers ──────────────────────────────────────────────────────────

def _content_hash(content: str) -> str:
    """SHA-256 of content text for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _lookup_lambda_base(conn: sqlite3.Connection, memory_type: str) -> float:
    """Return lambda_base for the given memory type (fallback: 0.16 for episodic)."""
    row = conn.execute(
        "SELECT lambda_base FROM lambda_base_constants WHERE memory_type = ?",
        (memory_type,),
    ).fetchone()
    return row["lambda_base"] if row else 0.16


def _compute_stability(importance: int) -> float:
    """Map importance [1,10] to stability [0,1] linearly."""
    return max(0.0, min(1.0, (importance - 1) / 9.0))


# ── Temporal scoring ──────────────────────────────────────────────────────────

def _temporal_score(
    access_count: int,
    last_accessed_at: Optional[float],
    created_at: float,
    decay_rate: float,
    stability: float,
) -> float:
    """Petrov ACT-R approximation + Ebbinghaus decay, combined.

    ACT-R Petrov: B_i = ln(n/(1-d)) - d*ln(t) with d=0.5
    Ebbinghaus:   strength = exp(-lambda_eff * t_days)
                  where lambda_eff = decay_rate * (1 - stability * 0.8)
    Combined:
        access_count == 0 -> strength * 0.5
        otherwise         -> actr_norm * 0.5 + strength * 0.5
    """
    now = time.time()
    ref_time = last_accessed_at if last_accessed_at else created_at
    t_days = max((now - ref_time) / 86400.0, 1e-6)

    lambda_eff = decay_rate * (1.0 - stability * 0.8)
    strength = math.exp(-lambda_eff * t_days)

    if access_count == 0:
        return strength * 0.5

    # ACT-R Petrov with d=0.5: B = ln(n/(1-d)) - d*ln(t_days)
    d = 0.5
    actr = math.log(access_count / (1.0 - d)) - d * math.log(t_days)
    # Normalise to [0,1] via linear clamp (ACT-R range empirically ~[-5, 5])
    actr_norm = max(0.0, min(1.0, (actr + 5.0) / 10.0))
    return actr_norm * 0.5 + strength * 0.5


# ── Public API ────────────────────────────────────────────────────────────────

def insert_memory(memory_dict: dict, session_id: str, project: str) -> str:
    """Insert a memory into the hub table and return its UUID.

    Skips exact content-hash duplicates within the same project (returns
    the existing id). Computes content_hash, decay_rate from lambda_base_constants,
    and stability from importance automatically.

    memory_dict recognised keys:
        content, memory_type, importance, extraction_confidence, confidence,
        subject, predicate, object, source_expert, event_time, valid_from,
        valid_to, decay_rate, stability, temporal_confidence
    """
    conn = get_db()
    now = time.time()

    content = memory_dict.get("content", "").strip()
    memory_type = memory_dict.get("memory_type") or memory_dict.get("type", "episodic")
    importance = int(memory_dict.get("importance", 5))
    content_hash = _content_hash(content)

    # Skip exact duplicates within the same project
    existing = conn.execute(
        "SELECT id FROM memories WHERE content_hash = ? AND project = ? AND superseded_by IS NULL",
        (content_hash, project),
    ).fetchone()
    if existing:
        conn.close()
        return existing["id"]

    memory_id = str(uuid.uuid4())
    lambda_base = _lookup_lambda_base(conn, memory_type)
    decay_rate = float(memory_dict.get("decay_rate", lambda_base))
    stability = float(memory_dict.get("stability", _compute_stability(importance)))
    gc_protected = 1 if importance >= 9 else 0

    conn.execute(
        """
        INSERT INTO memories (
            id, content, content_hash, memory_type, importance,
            extraction_confidence, confidence,
            subject, predicate, object,
            session_id, source_expert, project,
            created_at, event_time, valid_from, valid_to,
            last_accessed_at, access_count,
            superseded_by, superseded_at,
            decay_rate, stability, temporal_confidence,
            reinforcement_count, promotion_candidate,
            temporal_score, score_computed_at,
            gc_eligible, gc_protected
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            NULL, 0,
            NULL, NULL,
            ?, ?, ?,
            0, 0,
            NULL, NULL,
            0, ?
        )
        """,
        (
            memory_id, content, content_hash, memory_type, importance,
            float(memory_dict.get("extraction_confidence") or memory_dict.get("confidence", 0.8)),
            1.0,  # retrieval confidence starts at 1.0, reduced on contradiction
            memory_dict.get("subject"),
            memory_dict.get("predicate"),
            memory_dict.get("object"),
            session_id,
            memory_dict.get("source_expert"),
            project,
            now,
            memory_dict.get("event_time"),
            memory_dict.get("valid_from"),
            memory_dict.get("valid_to"),
            decay_rate,
            stability,
            float(memory_dict.get("temporal_confidence", 1.0)),
            gc_protected,
        ),
    )
    conn.commit()
    conn.close()
    return memory_id


def detect_supersession(
    new_memory_id: str,
    subject: Optional[str],
    predicate: Optional[str],
) -> Optional[str]:
    """Find the most recent active memory with the same subject+predicate.

    Marks it superseded_by new_memory_id, writes a supersession_events row,
    and returns the old memory's id. Returns None when no match found or when
    subject/predicate are absent.
    """
    if not subject or not predicate:
        return None

    conn = get_db()
    now = time.time()

    row = conn.execute(
        """
        SELECT id FROM memories
        WHERE subject = ?
          AND predicate = ?
          AND superseded_by IS NULL
          AND id != ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (subject, predicate, new_memory_id),
    ).fetchone()

    if not row:
        conn.close()
        return None

    old_id = row["id"]
    conn.execute(
        "UPDATE memories SET superseded_by = ?, superseded_at = ? WHERE id = ?",
        (new_memory_id, now, old_id),
    )
    conn.execute(
        """
        INSERT INTO supersession_events (
            old_memory_id, new_memory_id, event_time, detected_by,
            processed_by_temporal, processed_by_kg, processed_by_contextual,
            created_at
        ) VALUES (?, ?, ?, 'temporal', 0, 0, 0, ?)
        """,
        (old_id, new_memory_id, now, now),
    )
    conn.commit()
    conn.close()
    return old_id


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def detect_content_supersession(
    new_memory_id: str,
    new_content: str,
    new_memory_type: str,
    threshold: float = 0.6,
) -> Optional[str]:
    """Find existing active memories with similar content and supersede them.

    Uses word-level Jaccard similarity. If an existing memory of the same type
    has similarity >= threshold with the new content, the old one is superseded.
    Returns the superseded memory id, or None.
    """
    conn = get_db()
    now = time.time()

    rows = conn.execute(
        """
        SELECT id, content FROM memories
        WHERE memory_type = ?
          AND superseded_by IS NULL
          AND gc_eligible = 0
          AND id != ?
        ORDER BY created_at DESC
        LIMIT 50
        """,
        (new_memory_type, new_memory_id),
    ).fetchall()

    best_match_id = None
    best_similarity = 0.0

    for row in rows:
        sim = _jaccard_similarity(new_content, row["content"])
        if sim >= threshold and sim > best_similarity:
            best_similarity = sim
            best_match_id = row["id"]

    if not best_match_id:
        conn.close()
        return None

    conn.execute(
        "UPDATE memories SET superseded_by = ?, superseded_at = ? WHERE id = ?",
        (new_memory_id, now, best_match_id),
    )
    conn.execute(
        """
        INSERT INTO supersession_events (
            old_memory_id, new_memory_id, event_time, detected_by,
            processed_by_temporal, processed_by_kg, processed_by_contextual,
            created_at
        ) VALUES (?, ?, ?, 'content_similarity', 0, 0, 0, ?)
        """,
        (best_match_id, new_memory_id, now, now),
    )
    conn.commit()
    conn.close()
    return best_match_id


def get_memories_for_session_start(
    project: Optional[str] = None,
    min_importance: int = 7,
) -> list[dict]:
    """Return procedural + correction memories for session priming.

    Filters by importance >= min_importance, excludes superseded and gc_eligible
    rows. Applies Ebbinghaus + ACT-R temporal scoring and returns results ordered
    by temporal_score descending.
    """
    conn = get_db()
    params: list = [min_importance]
    project_clause = ""
    if project:
        # Prefix match: memories stored under /foo/bar/baz are found
        # when session starts in /foo/bar (parent directory)
        project_clause = "AND (project = ? OR project LIKE ? || '/%' OR ? LIKE project || '/%')"
        params.extend([project, project, project])

    rows = conn.execute(
        f"""
        SELECT id, content, memory_type, importance, subject, predicate,
               access_count, last_accessed_at, created_at, decay_rate,
               stability, session_id, project, reinforcement_count
        FROM memories
        WHERE memory_type IN ('procedural', 'correction')
          AND importance >= ?
          AND superseded_by IS NULL
          AND gc_eligible = 0
          {project_clause}
        ORDER BY importance DESC, created_at DESC
        LIMIT 200
        """,
        params,
    ).fetchall()

    results = []
    for row in rows:
        score = _temporal_score(
            access_count=row["access_count"],
            last_accessed_at=row["last_accessed_at"],
            created_at=row["created_at"],
            decay_rate=row["decay_rate"],
            stability=row["stability"],
        )
        d = dict(row)
        d["temporal_score"] = score
        results.append(d)

    conn.close()
    results.sort(key=lambda x: x["temporal_score"], reverse=True)
    return results


def record_session(session_id: str, started_at: float) -> None:
    """Insert a new session row (no-op if session_id already exists)."""
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
        (session_id, started_at),
    )
    conn.commit()
    conn.close()


def end_session(
    session_id: str,
    ended_at: float,
    summary: Optional[str] = None,
) -> None:
    """Update session with end time, computed duration, and optional summary."""
    conn = get_db()
    row = conn.execute(
        "SELECT started_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    duration = (ended_at - row["started_at"]) if row else None
    conn.execute(
        """
        UPDATE sessions
        SET ended_at = ?, duration_seconds = ?, summary = ?
        WHERE id = ?
        """,
        (ended_at, duration, summary, session_id),
    )
    conn.commit()
    conn.close()


def get_reinforcement_count(trigger_condition: str) -> int:
    """Count non-superseded memories whose content contains trigger_condition.

    Used for cross-session pattern detection. Returns 0 if trigger_condition
    is empty or no matches found.
    """
    if not trigger_condition:
        return 0

    conn = get_db()
    row = conn.execute(
        """
        SELECT COUNT(*) AS cnt FROM memories
        WHERE content LIKE ?
          AND superseded_by IS NULL
          AND gc_eligible = 0
        """,
        (f"%{trigger_condition}%",),
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0
