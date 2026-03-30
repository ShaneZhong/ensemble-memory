#!/usr/bin/env python3
"""db.py — Shared SQLite database module for the ensemble memory system.

All hooks import this module to access the temporal hub (memories table)
and supporting tables. Handles DB initialization, WAL mode, and the core
temporal scoring logic.

Env vars:
    ENSEMBLE_MEMORY_DIR   Root data directory (default: ~/.ensemble_memory)
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ensemble_memory.db")

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

-- Knowledge Graph tables
CREATE TABLE IF NOT EXISTS kg_entities (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    type                TEXT NOT NULL,
    subtype             TEXT,
    description         TEXT,
    aliases             TEXT,           -- JSON array
    first_seen          REAL NOT NULL,
    last_updated        REAL NOT NULL,
    last_accessed       REAL,
    access_count        INTEGER DEFAULT 0,
    session_count       INTEGER DEFAULT 0,
    weight              REAL DEFAULT 0.5,
    community_id        INTEGER,
    temporal_memory_id  TEXT,
    FOREIGN KEY (temporal_memory_id) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_kg_entity_name ON kg_entities(name);
CREATE INDEX IF NOT EXISTS idx_kg_entity_type ON kg_entities(type);
CREATE INDEX IF NOT EXISTS idx_kg_entity_community ON kg_entities(community_id);

CREATE TABLE IF NOT EXISTS kg_relationships (
    id                  TEXT PRIMARY KEY,
    subject_id          TEXT NOT NULL,
    predicate           TEXT NOT NULL
                        CHECK(predicate IN ('USES','DEPENDS_ON','CONFLICTS_WITH',
                              'HAS_CONSTRAINT','HAS_VERSION','RUNS_ON',
                              'APPLIES_TO','PREVENTS','SUPERSEDES','CAUSED_BY',
                              'IMPLEMENTED_IN','STORED_IN','PART_OF','AFFECTS',
                              'RELATED_TO','WORKS_ON')),
    object_id           TEXT NOT NULL,
    evidence            TEXT,
    confidence          REAL DEFAULT 0.5,
    valid_from          REAL,
    valid_until         REAL,
    episode_id          TEXT,
    temporal_memory_id  TEXT,
    synced_to_temporal  INTEGER DEFAULT 0,
    created_at          REAL NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES kg_entities(id),
    FOREIGN KEY (object_id) REFERENCES kg_entities(id),
    FOREIGN KEY (temporal_memory_id) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_kg_rel_subject ON kg_relationships(subject_id, valid_until);
CREATE INDEX IF NOT EXISTS idx_kg_rel_object ON kg_relationships(object_id, valid_until);
CREATE INDEX IF NOT EXISTS idx_kg_rel_predicate ON kg_relationships(predicate);
CREATE INDEX IF NOT EXISTS idx_kg_rel_valid ON kg_relationships(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_kg_rel_sync ON kg_relationships(synced_to_temporal)
    WHERE synced_to_temporal = 0;

CREATE TABLE IF NOT EXISTS kg_memory_links (
    id                  TEXT PRIMARY KEY,
    source_entity_id    TEXT NOT NULL,
    target_entity_id    TEXT NOT NULL,
    link_type           TEXT NOT NULL
                        CHECK(link_type IN ('RELATED','CONTRADICTS','SUPERSEDES',
                              'EVOLVED_FROM','SUPPORTS','REFINES','ENABLES',
                              'CAUSED_BY')),
    strength            REAL DEFAULT 0.5,
    created_at          REAL NOT NULL,
    FOREIGN KEY (source_entity_id) REFERENCES kg_entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES kg_entities(id)
);
CREATE INDEX IF NOT EXISTS idx_kg_memlink_source ON kg_memory_links(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_memlink_target ON kg_memory_links(target_entity_id);

CREATE TABLE IF NOT EXISTS kg_episodes (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    session_timestamp   REAL NOT NULL,
    event_time          REAL,
    content             TEXT,
    summary             TEXT
);

CREATE TABLE IF NOT EXISTS kg_appears_in (
    entity_id   TEXT NOT NULL,
    episode_id  TEXT NOT NULL,
    role        TEXT DEFAULT 'context',
    PRIMARY KEY (entity_id, episode_id),
    FOREIGN KEY (entity_id) REFERENCES kg_entities(id),
    FOREIGN KEY (episode_id) REFERENCES kg_episodes(id)
);
CREATE INDEX IF NOT EXISTS idx_kg_appears_entity ON kg_appears_in(entity_id);

CREATE TABLE IF NOT EXISTS kg_decay_config (
    predicate       TEXT PRIMARY KEY,
    decay_window_days INTEGER,
    description     TEXT
);

INSERT OR IGNORE INTO kg_decay_config VALUES
    ('HAS_VERSION',    30,   'Version info ages quickly'),
    ('USES',           180,  'Tech choices evolve slowly'),
    ('DEPENDS_ON',     180,  'Dependencies change slowly'),
    ('CONFLICTS_WITH', 90,   'Conflicts may be resolved'),
    ('HAS_CONSTRAINT', 180,  'Constraints are fairly stable'),
    ('RUNS_ON',        180,  'Runtime environment changes slowly'),
    ('APPLIES_TO',     NULL, 'Rules are permanent until superseded'),
    ('PREVENTS',       NULL, 'Prevention rules are permanent'),
    ('SUPERSEDES',     NULL, 'Supersession is a permanent record'),
    ('CAUSED_BY',      180,  'Causal links age with context'),
    ('IMPLEMENTED_IN', 180,  'Implementation details evolve'),
    ('STORED_IN',      180,  'Storage locations change'),
    ('PART_OF',        NULL, 'Structural relationships are stable'),
    ('AFFECTS',        90,   'Effect relationships may change'),
    ('RELATED_TO',     180,  'General relationships age'),
    ('WORKS_ON',       90,   'Work assignments change');

CREATE TABLE IF NOT EXISTS kg_sync_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
INSERT OR IGNORE INTO kg_sync_state VALUES
    ('last_claude_md_sync', '0', 0),
    ('last_memory_md_sync', '0', 0);

-- Phase 4: Decision Vault (lean structured index over memories)
CREATE TABLE IF NOT EXISTS decisions (
    id                TEXT PRIMARY KEY,
    memory_id         TEXT NOT NULL REFERENCES memories(id),
    decision_type     TEXT NOT NULL
                      CHECK(decision_type IN ('ARCHITECTURAL','PREFERENCE',
                            'ERROR_RESOLUTION','CONSTRAINT','PATTERN')),
    content_hash      TEXT NOT NULL,
    keywords          TEXT,          -- JSON array
    files_referenced  TEXT,          -- JSON array
    project           TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project);
CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_decisions_content_hash ON decisions(content_hash);
CREATE INDEX IF NOT EXISTS idx_decisions_memory_id ON decisions(memory_id);
"""

# ── Connection ─────────────────────────────────────────────────────────────────

_KG_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS kg_entities_fts USING fts5(
    name, aliases, description,
    content='kg_entities', content_rowid='rowid'
);
"""

_DECISIONS_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    text, keywords, decision_type,
    content='decisions', content_rowid='rowid',
    tokenize='porter ascii'
);
"""

_DECISIONS_TRIGGERS_DDL = """
CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions BEGIN
    INSERT INTO decisions_fts(rowid, text, keywords, decision_type)
    SELECT new.rowid,
           (SELECT content FROM memories WHERE id = new.memory_id),
           new.keywords,
           new.decision_type;
END;

CREATE TRIGGER IF NOT EXISTS decisions_ad AFTER DELETE ON decisions BEGIN
    INSERT INTO decisions_fts(decisions_fts, rowid, text, keywords, decision_type)
    VALUES ('delete', old.rowid,
            (SELECT content FROM memories WHERE id = old.memory_id),
            old.keywords,
            old.decision_type);
END;

CREATE TRIGGER IF NOT EXISTS decisions_au AFTER UPDATE ON decisions BEGIN
    INSERT INTO decisions_fts(decisions_fts, rowid, text, keywords, decision_type)
    VALUES ('delete', old.rowid,
            (SELECT content FROM memories WHERE id = old.memory_id),
            old.keywords,
            old.decision_type);
    INSERT INTO decisions_fts(rowid, text, keywords, decision_type)
    SELECT new.rowid,
           (SELECT content FROM memories WHERE id = new.memory_id),
           new.keywords,
           new.decision_type;
END;
"""


_db_initialized: set[str] = set()


def get_db() -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection, creating tables on first run."""
    db_path = str(_db_path())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # PRAGMAs are per-connection in SQLite — must run on every connection
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")

    if db_path not in _db_initialized:
        conn.executescript(_DDL)
        # FTS5 virtual table — wrapped in try/except for safety
        try:
            conn.executescript(_KG_FTS_DDL)
        except sqlite3.OperationalError:
            pass
        # Decisions FTS5 + triggers — Phase 4
        try:
            conn.executescript(_DECISIONS_FTS_DDL)
            conn.executescript(_DECISIONS_TRIGGERS_DDL)
        except sqlite3.OperationalError:
            pass
        conn.commit()
        _db_initialized.add(db_path)

    return conn


def ensure_embedding_column():
    """Add embedding column if it doesn't exist (migration for Phase 2)."""
    conn = get_db()
    try:
        conn.execute("SELECT embedding FROM memories LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        conn.commit()
    conn.close()


ensure_embedding_column()


def ensure_enrichment_columns():
    """Add enriched_text and enrichment_quality columns if missing (Phase 5 migration)."""
    conn = get_db()
    try:
        conn.execute("SELECT enriched_text FROM memories LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE memories ADD COLUMN enriched_text TEXT")
        conn.execute("ALTER TABLE memories ADD COLUMN enrichment_quality REAL DEFAULT 0.0")
        conn.commit()
    conn.close()


ensure_enrichment_columns()


def store_enrichment(memory_id: str, enriched_text: str, quality: float):
    """Store enriched text and quality score for a memory."""
    conn = get_db()
    conn.execute(
        "UPDATE memories SET enriched_text = ?, enrichment_quality = ? WHERE id = ?",
        (enriched_text, quality, memory_id),
    )
    conn.commit()
    conn.close()


def store_embedding(memory_id: str, embedding: list[float]):
    """Store embedding vector for a memory."""
    conn = get_db()
    conn.execute(
        "UPDATE memories SET embedding = ? WHERE id = ?",
        (json.dumps(embedding), memory_id)
    )
    conn.commit()
    conn.close()


def get_memories_with_embeddings(project=None, min_importance=1):
    """Get all active memories with their embeddings for similarity search."""
    conn = get_db()
    conn.row_factory = sqlite3.Row
    query = """
        SELECT id, content, memory_type, importance, embedding,
               created_at, access_count, decay_rate, stability
        FROM memories
        WHERE superseded_by IS NULL
          AND gc_eligible = 0
          AND importance >= ?
          AND embedding IS NOT NULL
    """
    params = [min_importance]
    if project:
        query += " AND (project = ? OR project LIKE ? || '/%' OR ? LIKE project || '/%')"
        params.extend([project, project, project])
    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        if d['embedding']:
            d['embedding'] = json.loads(d['embedding'])
        result.append(d)
    return result


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

def temporal_score(
    access_count: int,
    last_accessed_at: Optional[float],
    created_at: float,
    decay_rate: float,
    stability: float,
    d: float = 0.5,
) -> float:
    """Canonical temporal score computation (Ebbinghaus + ACT-R Petrov hybrid).

    Single source of truth — used by both daemon and batch computation.

    ACT-R Petrov: B_i = ln(n/(1-d)) - d*ln(t) with d=0.5
    Ebbinghaus:   strength = exp(-lambda_eff * t_days)
                  where lambda_eff = decay_rate * (1 - stability * 0.8)
    Combined:
        access_count == 0 -> strength * 0.5
        otherwise         -> actr_norm * 0.5 + strength * 0.5

    Returns score in [0, 1].
    """
    now = time.time()
    ref_time = last_accessed_at if last_accessed_at else created_at
    t_days = max((now - ref_time) / 86400.0, 1e-6)

    lambda_eff = decay_rate * (1.0 - stability * 0.8)
    strength = math.exp(-lambda_eff * t_days)

    if access_count == 0:
        return strength * 0.5

    # ACT-R Petrov with d=0.5: B = ln(n/(1-d)) - d*ln(t_days)
    actr = math.log(access_count / (1.0 - d)) - d * math.log(t_days)
    # Normalise to [0,1] via linear clamp (ACT-R range empirically ~[-5, 5])
    actr_norm = max(0.0, min(1.0, (actr + 5.0) / 10.0))
    return actr_norm * 0.5 + strength * 0.5


# Backward-compatible alias for internal callers
_temporal_score = temporal_score


def compute_temporal_scores(chunk_size: int = 100) -> int:
    """Batch-compute and cache temporal_score for all active memories.

    Processes in chunks of chunk_size to keep write locks short.
    Updates score_computed_at to prevent redundant computation.

    Returns the number of memories updated.
    """
    conn = get_db()
    now = time.time()
    updated = 0

    try:
        while True:
            rows = conn.execute("""
                SELECT id, access_count, last_accessed_at, created_at,
                       decay_rate, stability
                FROM memories
                WHERE superseded_by IS NULL
                  AND gc_eligible = 0
                LIMIT ?
                OFFSET ?
            """, (chunk_size, updated)).fetchall()

            if not rows:
                break

            for row in rows:
                score = temporal_score(
                    access_count=row["access_count"],
                    last_accessed_at=row["last_accessed_at"],
                    created_at=row["created_at"],
                    decay_rate=row["decay_rate"],
                    stability=row["stability"],
                )
                conn.execute(
                    "UPDATE memories SET temporal_score = ?, score_computed_at = ? WHERE id = ?",
                    (score, now, row["id"]),
                )

            conn.commit()
            updated += len(rows)

            if len(rows) < chunk_size:
                break
    finally:
        conn.close()

    return updated


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
    try:
        now = time.time()

        content = memory_dict.get("content", "").strip()
        raw_type = memory_dict.get("memory_type") or memory_dict.get("type", "episodic")

        # Normalize memory_type: LLMs sometimes return "correction | semantic | procedural".
        # Split on '|' and ',' and pick the first token that is a valid memory_type.
        _VALID_MEMORY_TYPES = frozenset(["episodic", "semantic", "procedural", "correction"])
        memory_type = "episodic"  # fallback default
        for _part in (raw_type or "").replace(",", "|").split("|"):
            _candidate = _part.strip().lower()
            if _candidate in _VALID_MEMORY_TYPES:
                memory_type = _candidate
                break

        importance = int(memory_dict.get("importance", 5))
        content_hash = _content_hash(content)

        # Skip exact duplicates within the same project
        existing = conn.execute(
            "SELECT id FROM memories WHERE content_hash = ? AND project = ? AND superseded_by IS NULL",
            (content_hash, project),
        ).fetchone()
        if existing:
            return existing["id"]

        # Near-dedup: Jaccard similarity check against recent same-type memories
        near_dupes = conn.execute(
            """
            SELECT id, content FROM memories
            WHERE memory_type = ?
              AND project = ?
              AND superseded_by IS NULL
              AND gc_eligible = 0
              AND id != ?
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (memory_type, project, ""),  # empty string won't match any UUID
        ).fetchall()

        for candidate in near_dupes:
            if _jaccard_similarity(content, candidate["content"]) >= 0.85:
                # Near-duplicate found — update access time instead of inserting
                conn.execute(
                    "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                    (now, candidate["id"]),
                )
                conn.commit()
                return candidate["id"]

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
        return memory_id
    finally:
        conn.close()


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
    new_embedding: Optional[list[float]] = None,
) -> Optional[str]:
    """Find existing active memories with similar content and supersede them.

    If new_embedding is provided and candidates have embeddings, uses cosine
    similarity with threshold 0.85. Falls back to word-level Jaccard similarity
    with the supplied threshold (default 0.6) when embeddings are unavailable.
    Returns the superseded memory id, or None.
    """
    conn = get_db()
    now = time.time()

    rows = conn.execute(
        """
        SELECT id, content, embedding FROM memories
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
        candidate_embedding_raw = row["embedding"]
        # Use cosine similarity when both sides have embeddings
        if new_embedding and candidate_embedding_raw:
            candidate_embedding = json.loads(candidate_embedding_raw)
            try:
                import embeddings as _emb
                sim = _emb.cosine_similarity(new_embedding, candidate_embedding)
            except ImportError:
                sim = _jaccard_similarity(new_content, row["content"])
            cosine_threshold = 0.85
            if sim >= cosine_threshold and sim > best_similarity:
                best_similarity = sim
                best_match_id = row["id"]
        else:
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

    DEPRECATED: Use get_reinforcement_match() instead.
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


def get_reinforcement_match(
    match_text: str,
    memory_type: Optional[str] = None,
    subject: str = "",
    predicate: str = "",
    obj: str = "",
    new_embedding: Optional[list[float]] = None,
) -> tuple[int, Optional[str]]:
    """Find reinforcement match using structured triples, then embedding similarity.

    Three strategies in priority order:
    1. Exact triple match (same subject + predicate + object) — most reliable
    2. Embedding cosine similarity >= 0.90 — catches rephrased versions
    3. Exact LIKE substring match on content — final fallback

    Returns (reinforcement_count + 1, memory_id), or (0, None) if no match.
    """
    if not match_text and not subject:
        return 0, None

    conn = get_db()
    type_filter = "AND memory_type = ?" if memory_type else "AND memory_type IN ('procedural', 'correction')"
    type_params = [memory_type] if memory_type else []

    try:
        # Strategy 1: exact triple match (same subject+predicate+object = reinforcement)
        if subject and predicate and obj:
            row = conn.execute(
                f"""
                SELECT id, reinforcement_count FROM memories
                WHERE subject = ? AND predicate = ? AND object = ?
                  {type_filter}
                  AND superseded_by IS NULL
                  AND gc_eligible = 0
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [subject, predicate, obj] + type_params,
            ).fetchone()
            if row:
                return row["reinforcement_count"] + 1, row["id"]

        # Strategy 2: embedding cosine similarity >= 0.90
        # Note: store_memory.py does not currently pass new_embedding (embedding
        # is generated after reinforcement check). This strategy is available for
        # callers that have embeddings (e.g., daemon background promotion scan).
        if new_embedding:
            rows = conn.execute(
                f"""
                SELECT id, content, embedding, reinforcement_count FROM memories
                WHERE embedding IS NOT NULL
                  {type_filter}
                  AND superseded_by IS NULL
                  AND gc_eligible = 0
                ORDER BY created_at DESC
                LIMIT 50
                """,
                type_params,
            ).fetchall()
            try:
                import embeddings as _emb
                for row in rows:
                    try:
                        cand_emb = json.loads(row["embedding"])
                        sim = _emb.cosine_similarity(new_embedding, cand_emb)
                        if sim >= 0.90:
                            return row["reinforcement_count"] + 1, row["id"]
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue  # Skip corrupted embedding
            except ImportError:
                logger.debug("[reinforcement] embeddings module unavailable, skipping cosine strategy")

        # Strategy 3: exact LIKE substring match (final fallback)
        if match_text:
            escaped = match_text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            row = conn.execute(
                f"""
                SELECT id, reinforcement_count FROM memories
                WHERE content LIKE ? ESCAPE '\\'
                  {type_filter}
                  AND superseded_by IS NULL
                  AND gc_eligible = 0
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [f"%{escaped}%"] + type_params,
            ).fetchone()
            if row:
                return row["reinforcement_count"] + 1, row["id"]

        return 0, None
    finally:
        conn.close()


def increment_reinforcement(memory_id: str) -> int:
    """Increment reinforcement_count and update stability for a memory.

    Stability rules:
      count=2: stability += 0.1
      count=3: stability += 0.2, set promotion_candidate=1
      count>=5: stability = 1.0 (permanent)

    Returns the new reinforcement_count, or 0 if memory not found.
    """
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT reinforcement_count, importance FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if not row:
            return 0

        old_count = row["reinforcement_count"]
        new_count = old_count + 1
        importance = row["importance"]

        # Compute base stability from importance
        base_stability = _compute_stability(importance)

        # Apply reinforcement bonuses
        if new_count >= 5:
            new_stability = 1.0
        elif new_count >= 3:
            new_stability = min(1.0, base_stability + 0.2)
        elif new_count >= 2:
            new_stability = min(1.0, base_stability + 0.1)
        else:
            new_stability = base_stability

        promotion = 1 if new_count >= 3 else 0

        conn.execute(
            """
            UPDATE memories
            SET reinforcement_count = ?,
                stability = ?,
                promotion_candidate = ?,
                last_accessed_at = ?
            WHERE id = ?
            """,
            (new_count, new_stability, promotion, time.time(), memory_id),
        )
        conn.commit()
        return new_count
    finally:
        conn.close()


def decay_stale_importance(days_threshold: int = 30, floor: int = 3) -> int:
    """Decay importance by 1 for memories not accessed in `days_threshold` days.

    Uses last_accessed_at if set, otherwise created_at. Floors at `floor`.
    Returns the number of memories decayed.

    Idempotency: only runs once per 23 hours to avoid repeated decay on
    daemon restarts.
    """
    conn = get_db()
    now = time.time()

    # Idempotency: only run once per 23 hours
    last_run = conn.execute(
        "SELECT value FROM kg_sync_state WHERE key = 'last_decay_run'"
    ).fetchone()
    if last_run:
        last_ts = float(last_run["value"])
        if (now - last_ts) < 82800:  # 23 hours in seconds
            conn.close()
            return 0

    cutoff = now - (days_threshold * 86400)

    cursor = conn.execute(
        """
        UPDATE memories
        SET importance = importance - 1
        WHERE importance > ?
          AND superseded_by IS NULL
          AND gc_eligible = 0
          AND COALESCE(last_accessed_at, created_at) < ?
        """,
        (floor, cutoff),
    )
    count = cursor.rowcount

    # Record this run
    conn.execute(
        "INSERT OR REPLACE INTO kg_sync_state (key, value, updated_at) VALUES ('last_decay_run', ?, ?)",
        (str(now), now),
    )
    conn.commit()
    conn.close()
    return count


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
            return None  # Not a valid decision — skip

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
    import re
    tokens = re.findall(r'[a-zA-Z0-9_]+', query)
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


# NOTE: Daemon _bm25_search has a similar FTS5 query. Keep them in sync.
def search_decisions_bm25(query: str, project: str = "", limit: int = 10) -> list[dict]:
    """Search decisions using FTS5 BM25 ranking. Returns list of dicts."""
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


# ── Phase 6 Sprint 2: Supersession Event Bus + Chain Depth Pruning ────────


def _process_supersession_kg(conn: sqlite3.Connection, old_memory_id: str, new_memory_id: str):
    """KG expert: set valid_until on relationships directly tied to old memory.

    Only invalidates relationships that reference the old memory's subject entity
    AND were linked to the old memory's session (via episode). Does NOT blanket-
    invalidate all relationships for the entity.
    """
    now = time.time()
    old_mem = conn.execute(
        "SELECT session_id, subject FROM memories WHERE id = ?", (old_memory_id,)
    ).fetchone()
    if not old_mem or not old_mem["subject"]:
        return

    subject = old_mem["subject"]
    entity = conn.execute(
        "SELECT id FROM kg_entities WHERE name = ?", (subject,)
    ).fetchone()
    if not entity:
        return

    # Only invalidate relationships from the same session as the old memory
    # to avoid blanket-invalidating unrelated relationships for the same entity.
    session_episodes = conn.execute(
        "SELECT id FROM kg_episodes WHERE session_id = ?", (old_mem["session_id"],)
    ).fetchall()
    episode_ids = [e["id"] for e in session_episodes]

    if episode_ids:
        placeholders = ",".join("?" * len(episode_ids))
        conn.execute(
            f"""
            UPDATE kg_relationships
            SET valid_until = ?
            WHERE (subject_id = ? OR object_id = ?)
              AND valid_until IS NULL
              AND episode_id IN ({placeholders})
            """,
            [now, entity["id"], entity["id"]] + episode_ids,
        )
    else:
        # Fallback: if no episodes found, invalidate relationships created
        # around the same time as the old memory (within 1 hour)
        old_created = conn.execute(
            "SELECT created_at FROM memories WHERE id = ?", (old_memory_id,)
        ).fetchone()
        if old_created:
            window = 3600  # 1 hour
            conn.execute(
                """
                UPDATE kg_relationships
                SET valid_until = ?
                WHERE (subject_id = ? OR object_id = ?)
                  AND valid_until IS NULL
                  AND created_at BETWEEN ? AND ?
                """,
                (now, entity["id"], entity["id"],
                 old_created["created_at"] - window, old_created["created_at"] + window),
            )


def _process_supersession_contextual(conn: sqlite3.Connection, old_memory_id: str):
    """Contextual expert: clear enriched_text on superseded memory."""
    conn.execute(
        "UPDATE memories SET enriched_text = NULL, enrichment_quality = 0.0 WHERE id = ?",
        (old_memory_id,),
    )


def process_supersession_events() -> dict:
    """Process unprocessed supersession events across all three experts.

    Per-event, per-expert processing: only processes experts where
    processed_by_X = 0. Uses single connection per event for atomicity.

    Returns dict with counts: {'processed': N, 'temporal': N, 'kg': N, 'contextual': N}
    """
    conn = get_db()
    stats = {'processed': 0, 'temporal': 0, 'kg': 0, 'contextual': 0}

    try:
        events = conn.execute("""
            SELECT id, old_memory_id, new_memory_id,
                   processed_by_temporal, processed_by_kg, processed_by_contextual
            FROM supersession_events
            WHERE processed_by_temporal = 0
               OR processed_by_kg = 0
               OR processed_by_contextual = 0
        """).fetchall()

        for event in events:
            event_id = event["id"]
            old_id = event["old_memory_id"]
            new_id = event["new_memory_id"]
            stats['processed'] += 1

            # Temporal processor
            if event["processed_by_temporal"] == 0:
                try:
                    # superseded_by already set at detection time
                    # Just mark as processed
                    conn.execute(
                        "UPDATE supersession_events SET processed_by_temporal = 1 WHERE id = ?",
                        (event_id,),
                    )
                    stats['temporal'] += 1
                except Exception as exc:
                    logger.warning("[event_bus] Temporal processor failed for event %s: %s", event_id, exc)

            # KG processor
            if event["processed_by_kg"] == 0:
                try:
                    _process_supersession_kg(conn, old_id, new_id)
                    conn.execute(
                        "UPDATE supersession_events SET processed_by_kg = 1 WHERE id = ?",
                        (event_id,),
                    )
                    stats['kg'] += 1
                except Exception as exc:
                    logger.warning("[event_bus] KG processor failed for event %s: %s", event_id, exc)

            # Contextual processor
            if event["processed_by_contextual"] == 0:
                try:
                    _process_supersession_contextual(conn, old_id)
                    conn.execute(
                        "UPDATE supersession_events SET processed_by_contextual = 1 WHERE id = ?",
                        (event_id,),
                    )
                    stats['contextual'] += 1
                except Exception as exc:
                    logger.warning("[event_bus] Contextual processor failed for event %s: %s", event_id, exc)

            conn.commit()
    finally:
        conn.close()

    return stats


def get_supersession_chain(memory_id: str) -> list[str]:
    """Walk the supersession chain from a memory to its newest successor.

    Returns list of memory IDs from oldest (memory_id first) following
    superseded_by pointers. Handles cycles and missing memories gracefully.
    """
    conn = get_db()
    chain = [memory_id]
    seen = {memory_id}
    current = memory_id

    try:
        while True:
            row = conn.execute(
                "SELECT superseded_by FROM memories WHERE id = ?", (current,)
            ).fetchone()
            if not row or not row["superseded_by"]:
                break
            next_id = row["superseded_by"]
            if next_id in seen:
                break  # Cycle detection
            seen.add(next_id)
            chain.append(next_id)
            current = next_id
    finally:
        conn.close()

    return chain


def enforce_chain_depth_limits() -> int:
    """Walk all supersession chains and mark excess memories as gc_eligible.

    Uses supersession_depth_limits table for per-type max chain depth.
    Protected memories (importance >= 9) are never GC'd.

    Returns the number of memories marked as gc_eligible.
    """
    conn = get_db()
    marked = 0

    try:
        # Get depth limits
        limits = {}
        for row in conn.execute("SELECT memory_type, max_chain_depth FROM supersession_depth_limits").fetchall():
            limits[row["memory_type"]] = row["max_chain_depth"]

        # Find all memories that have been superseded (chain roots = oldest links)
        roots = conn.execute("""
            SELECT DISTINCT m.id, m.memory_type
            FROM memories m
            WHERE m.superseded_by IS NOT NULL
              AND m.gc_eligible = 0
              AND m.gc_protected = 0
        """).fetchall()

        # Track which memories we've already processed in some chain
        processed_chains = set()

        for root in roots:
            mem_id = root["id"]
            if mem_id in processed_chains:
                continue

            # Build chain: walk via superseded_by pointers from oldest to newest
            chain = []
            current = mem_id
            seen = set()
            while current:
                if current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = conn.execute(
                    "SELECT superseded_by FROM memories WHERE id = ?", (current,)
                ).fetchone()
                if not row or not row["superseded_by"]:
                    break
                current = row["superseded_by"]

            processed_chains.update(chain)

            # Check depth limit
            mem_type = root["memory_type"]
            max_depth = limits.get(mem_type, 5)  # default 5

            if len(chain) > max_depth:
                # Mark excess oldest links as gc_eligible
                # chain[0] is the oldest; chain[-1] is the newest
                # Keep the newest max_depth entries, mark the rest
                excess = chain[:len(chain) - max_depth]
                for excess_id in excess:
                    row = conn.execute(
                        "SELECT gc_protected, gc_eligible, importance FROM memories WHERE id = ?",
                        (excess_id,),
                    ).fetchone()
                    if row and row["gc_protected"] == 0 and row["gc_eligible"] == 0 and row["importance"] < 9:
                        conn.execute(
                            "UPDATE memories SET gc_eligible = 1 WHERE id = ?",
                            (excess_id,),
                        )
                        marked += 1

        conn.commit()
    finally:
        conn.close()

    return marked


def run_garbage_collection() -> dict:
    """Soft-delete memories eligible for garbage collection.

    Two eligibility paths (OR):
    1. Chain-pruned: gc_eligible = 1 AND gc_protected = 0
    2. Naturally forgotten: temporal_score < 0.005 AND superseded_by IS NOT NULL AND gc_protected = 0

    Protected memories (importance >= 9) are never GC'd.
    Does NOT delete rows — sets gc_eligible = 1 for exclusion from retrieval.

    Returns {'gc_chain_pruned': N, 'gc_forgotten': N, 'total': N}
    All counts are delta (newly GC'd this run), not cumulative.
    """
    conn = get_db()
    stats = {'gc_chain_pruned': 0, 'gc_forgotten': 0, 'total': 0}

    try:
        # Path 2: Naturally forgotten + superseded (mark newly eligible)
        cursor = conn.execute("""
            UPDATE memories
            SET gc_eligible = 1
            WHERE temporal_score IS NOT NULL
              AND temporal_score < 0.005
              AND superseded_by IS NOT NULL
              AND gc_eligible = 0
              AND gc_protected = 0
              AND importance < 9
        """)
        stats['gc_forgotten'] = cursor.rowcount

        conn.commit()
        # Total is only newly GC'd this run (chain pruning is counted
        # by enforce_chain_depth_limits, not double-counted here)
        stats['total'] = stats['gc_forgotten']
    finally:
        conn.close()

    return stats
