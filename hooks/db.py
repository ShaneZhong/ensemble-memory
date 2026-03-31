#!/usr/bin/env python3
"""db.py — Facade module for the ensemble memory database layer.

Provides get_db() connection management, DDL schema, and re-exports all public
functions from sub-modules (db_memory, db_lifecycle, db_decisions) so that
existing callers continue to work unchanged via `import db; db.insert_memory(...)`.

Env vars:
    ENSEMBLE_MEMORY_DIR   Root data directory (default: ~/.ensemble_memory)
"""

import logging
import os
import sqlite3
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

-- Phase 7: A-MEM memory-to-memory evolution links
CREATE TABLE IF NOT EXISTS amem_memory_links (
    id                  TEXT PRIMARY KEY,
    source_memory_id    TEXT NOT NULL REFERENCES memories(id),
    target_memory_id    TEXT NOT NULL REFERENCES memories(id),
    link_type           TEXT NOT NULL
                        CHECK(link_type IN ('RELATED','CONTRADICTS','SUPERSEDES',
                              'EVOLVED_FROM','SUPPORTS','REFINES','ENABLES',
                              'CAUSED_BY')),
    strength            REAL DEFAULT 0.5,
    created_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_amem_link_source ON amem_memory_links(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_amem_link_target ON amem_memory_links(target_memory_id);

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


# ── Re-export sub-module APIs ─────────────────────────────────────────────────
# All public functions are re-exported so that `import db; db.insert_memory(...)`
# continues to work unchanged for all existing callers.

from db_memory import (  # noqa: E402, F401
    ensure_embedding_column,
    ensure_enrichment_columns,
    store_enrichment,
    store_embedding,
    get_memories_with_embeddings,
    insert_memory,
    detect_supersession,
    detect_content_supersession,
    get_memories_for_session_start,
    get_recent_context,
    record_session,
    end_session,
    temporal_score,
    _temporal_score,
    _compute_stability,
    _content_hash,
    _lookup_lambda_base,
    _jaccard_similarity,
)

from db_lifecycle import (  # noqa: E402, F401
    insert_memory_link,
    get_memory_links,
    queue_amem_evolution,
    get_pending_amem_queue,
    dequeue_amem,
    get_reinforcement_count,
    get_reinforcement_match,
    increment_reinforcement,
    decay_stale_importance,
    process_supersession_events,
    _process_supersession_kg,
    _process_supersession_contextual,
    get_supersession_chain,
    enforce_chain_depth_limits,
    run_garbage_collection,
    compute_temporal_scores,
)

from db_decisions import (  # noqa: E402, F401
    _VALID_DECISION_TYPES,
    insert_decision,
    search_decisions_bm25,
    _sanitize_fts5_query,
)


# ── Module-level migrations (run on import) ──────────────────────────────────
ensure_embedding_column()
ensure_enrichment_columns()
