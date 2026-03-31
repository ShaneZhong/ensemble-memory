#!/usr/bin/env python3
"""db_memory.py — Memory CRUD, embedding, enrichment, supersession detection, and temporal scoring.

Split from db.py for maintainability. All functions import get_db from db.
"""

import hashlib
import json
import logging
import math
import sqlite3
import time
import uuid
from typing import Optional

logger = logging.getLogger("ensemble_memory.db")


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


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


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


# ── Schema migrations ────────────────────────────────────────────────────────

def ensure_embedding_column():
    """Add embedding column if it doesn't exist (migration for Phase 2)."""
    from db import get_db
    conn = get_db()
    try:
        conn.execute("SELECT embedding FROM memories LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        conn.commit()
    conn.close()


def ensure_enrichment_columns():
    """Add enriched_text and enrichment_quality columns if missing (Phase 5 migration)."""
    from db import get_db
    conn = get_db()
    try:
        conn.execute("SELECT enriched_text FROM memories LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE memories ADD COLUMN enriched_text TEXT")
        conn.execute("ALTER TABLE memories ADD COLUMN enrichment_quality REAL DEFAULT 0.0")
        conn.commit()
    conn.close()


# ── Embedding migration ──────────────────────────────────────────────────────

def reembed_all_memories(batch_size: int = 32) -> int:
    """Re-embed all memories using the current embedding model.

    Used when upgrading embedding models (e.g., MiniLM 384-dim → BGE-M3 1024-dim).
    Processes in batches for efficiency. Returns count of re-embedded memories.
    """
    try:
        import embeddings
    except ImportError:
        logger.warning("embeddings module not available, cannot re-embed")
        return 0

    from db import get_db
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, content FROM memories
            WHERE superseded_by IS NULL AND gc_eligible = 0
            ORDER BY created_at DESC
            """,
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("No memories to re-embed")
        return 0

    total = len(rows)
    logger.info("Re-embedding %d memories with model: %s", total, embeddings.MODEL_NAME)
    reembedded = 0

    # Process in batches
    for i in range(0, total, batch_size):
        batch = rows[i:i + batch_size]
        texts = [row["content"] for row in batch]
        ids = [row["id"] for row in batch]

        try:
            vectors = embeddings.get_embeddings(texts)
            if vectors is None:
                logger.warning("Embedding model unavailable, stopping re-embed at %d/%d", reembedded, total)
                break

            conn = get_db()
            try:
                for mem_id, vec in zip(ids, vectors):
                    if vec is not None:
                        conn.execute(
                            "UPDATE memories SET embedding = ? WHERE id = ?",
                            (json.dumps(vec), mem_id),
                        )
                        reembedded += 1
                conn.commit()
            finally:
                conn.close()

            if (i + batch_size) % 100 < batch_size:
                logger.info("Re-embed progress: %d/%d", min(i + batch_size, total), total)

        except Exception as exc:
            logger.warning("Re-embed batch failed at offset %d: %s", i, exc)
            continue

    logger.info("Re-embedding complete: %d/%d memories updated", reembedded, total)
    return reembedded


# ── Public API ────────────────────────────────────────────────────────────────

def store_enrichment(memory_id: str, enriched_text: str, quality: float):
    """Store enriched text and quality score for a memory."""
    from db import get_db
    conn = get_db()
    conn.execute(
        "UPDATE memories SET enriched_text = ?, enrichment_quality = ? WHERE id = ?",
        (enriched_text, quality, memory_id),
    )
    conn.commit()
    conn.close()


def store_embedding(memory_id: str, embedding: list[float]):
    """Store embedding vector for a memory."""
    from db import get_db
    conn = get_db()
    conn.execute(
        "UPDATE memories SET embedding = ? WHERE id = ?",
        (json.dumps(embedding), memory_id)
    )
    conn.commit()
    conn.close()


def get_memories_with_embeddings(project=None, min_importance=1):
    """Get all active memories with their embeddings for similarity search."""
    from db import get_db
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
    from db import get_db
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

    from db import get_db
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
    from db import get_db
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

    Filters by importance >= min_importance, excludes superseded, gc_eligible,
    expired (valid_to < now), and future (valid_from > now) memories.
    Applies Ebbinghaus + ACT-R temporal scoring and returns results ordered
    by temporal_score descending.
    """
    import time as _time

    from db import get_db
    conn = get_db()
    now = _time.time()
    params: list = [min_importance, now, now]
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
          AND (valid_to IS NULL OR valid_to > ?)
          AND (valid_from IS NULL OR valid_from <= ?)
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


def get_recent_context(
    project: Optional[str] = None,
    limit: int = 30,
    min_importance: int = 5,
    exclude_ids: Optional[set] = None,
) -> list[dict]:
    """Return recent semantic + episodic memories for context injection.

    Complements get_memories_for_session_start() by providing recent non-rule
    memories. Excludes IDs already loaded as standing rules to avoid duplication.
    """
    import time as _time

    from db import get_db
    conn = get_db()
    now = _time.time()
    params: list = [min_importance, now, now]
    project_clause = ""
    if project:
        project_clause = "AND (project = ? OR project LIKE ? || '/%' OR ? LIKE project || '/%')"
        params.extend([project, project, project])
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT id, content, memory_type, importance, subject, created_at
        FROM memories
        WHERE memory_type IN ('semantic', 'episodic')
          AND importance >= ?
          AND superseded_by IS NULL
          AND gc_eligible = 0
          AND (valid_to IS NULL OR valid_to > ?)
          AND (valid_from IS NULL OR valid_from <= ?)
          {project_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        if exclude_ids and row["id"] in exclude_ids:
            continue
        results.append(dict(row))
    return results


def record_session(session_id: str, started_at: float) -> None:
    """Insert a new session row (no-op if session_id already exists)."""
    from db import get_db
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
    from db import get_db
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
