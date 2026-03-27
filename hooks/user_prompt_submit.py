#!/usr/bin/env python3
"""user_prompt_submit.py — Query-time memory retrieval for ensemble memory system.

Fires on every user prompt. Finds memories semantically relevant to the
incoming message and injects them as additionalContext so Claude sees
corrections and rules before it begins generating a response.

Usage: user_prompt_submit.py <message_text> [session_id]

Env vars:
    ENSEMBLE_MEMORY_PROJECT   Project path for scoping (default: cwd)
    ENSEMBLE_MEMORY_DIR       Root data dir (default: ~/.ensemble_memory)

Output:
    {"additionalContext": "...formatted memories..."} — when relevant memories found
    {}                                                 — when nothing relevant found
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Locate hooks dir and make siblings importable ────────────────────────────
HOOKS_DIR = Path(__file__).resolve().parent
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))

# ── Config ────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.20   # Cosine similarity floor for semantic retrieval (low = high recall, memories are pre-filtered by importance)
KEYWORD_THRESHOLD    = 0.15   # Jaccard word-overlap floor for fallback retrieval
TOP_K                = 5      # Maximum memories to inject per prompt
MIN_TEMPORAL_SCORE   = 0.05   # Skip memories with near-zero temporal relevance
MIN_IMPORTANCE       = 3      # Only retrieve memories at or above this importance

# Memory types allowed for query-time retrieval (all types, unlike session_start)
RETRIEVABLE_TYPES = {"procedural", "correction", "semantic", "episodic"}

# ── Session-scoped embedding cache ───────────────────────────────────────────
# Keyed by memory id → embedding vector. Populated on first call, reused
# across subsequent prompts within the same process (same Claude session).
_embedding_cache: dict[str, list[float]] = {}
_memories_cache: Optional[list[dict]] = None   # raw memory rows, loaded once
_cache_loaded_at: float = 0.0
_CACHE_TTL_SECONDS = 300  # re-query DB at most every 5 minutes

TYPE_LABEL = {
    "correction": "correction",
    "procedural": "procedural",
    "semantic":   "fact",
    "episodic":   "context",
}


# ── Embedding column migration ────────────────────────────────────────────────

def _ensure_embedding_column(conn) -> None:
    """Add `embedding` BLOB column to memories table if not present (idempotent)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    if "embedding" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        conn.commit()


def _backfill_embeddings(conn, get_embedding_fn) -> int:
    """Generate and store embeddings for memories that don't have one yet.

    Returns the number of memories backfilled. Runs only when sentence-transformers
    is available. Skips silently if not.
    """
    rows = conn.execute(
        """
        SELECT id, content FROM memories
        WHERE embedding IS NULL
          AND superseded_by IS NULL
          AND gc_eligible = 0
        LIMIT 500
        """
    ).fetchall()

    if not rows:
        return 0

    count = 0
    for row in rows:
        try:
            vec = get_embedding_fn(row["content"])
            if vec is not None:
                conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (json.dumps(vec), row["id"]),
                )
                count += 1
        except Exception:
            continue

    if count:
        conn.commit()
    return count


# ── Memory loading ────────────────────────────────────────────────────────────

def _load_memories(conn, project: str) -> list[dict]:
    """Load all active memories from SQLite, scoped by project prefix."""
    global _memories_cache, _cache_loaded_at

    now = time.time()
    if _memories_cache is not None and (now - _cache_loaded_at) < _CACHE_TTL_SECONDS:
        return _memories_cache

    # Project prefix matching mirrors get_memories_for_session_start in db.py
    params: list = [MIN_IMPORTANCE]
    project_clause = ""
    if project:
        project_clause = (
            "AND (project = ? OR project LIKE ? || '/%' OR ? LIKE project || '/%')"
        )
        params.extend([project, project, project])

    rows = conn.execute(
        f"""
        SELECT id, content, memory_type, importance, subject,
               access_count, last_accessed_at, created_at,
               decay_rate, stability, embedding
        FROM memories
        WHERE superseded_by IS NULL
          AND gc_eligible = 0
          AND importance >= ?
          {project_clause}
        ORDER BY importance DESC, created_at DESC
        LIMIT 500
        """,
        params,
    ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        raw_emb = d.pop("embedding", None)
        if raw_emb:
            try:
                d["embedding"] = json.loads(raw_emb)
            except (json.JSONDecodeError, TypeError):
                d["embedding"] = None
        else:
            d["embedding"] = None
        result.append(d)

    _memories_cache = result
    _cache_loaded_at = now
    return result


# ── Temporal scoring (mirrors db.py — imported directly if possible) ──────────

def _temporal_score(row: dict) -> float:
    import math
    now = time.time()
    last = row.get("last_accessed_at") or row.get("created_at", now)
    t_days = max((now - last) / 86400.0, 1e-6)

    stability = float(row.get("stability", 0.0))
    decay_rate = float(row.get("decay_rate", 0.16))
    lambda_eff = decay_rate * (1.0 - stability * 0.8)
    strength = math.exp(-lambda_eff * t_days)

    access_count = int(row.get("access_count", 0))
    if access_count == 0:
        return strength * 0.5

    d = 0.5
    import math as _m
    actr = _m.log(access_count / (1.0 - d)) - d * _m.log(t_days)
    actr_norm = max(0.0, min(1.0, (actr + 5.0) / 10.0))
    return actr_norm * 0.5 + strength * 0.5


# ── Keyword fallback (no sentence-transformers) ───────────────────────────────

def _keyword_similarity(query: str, content: str) -> float:
    """Word-overlap Jaccard similarity — fast fallback when embeddings unavailable."""
    STOPWORDS = {
        "the", "a", "an", "is", "in", "it", "to", "and", "or", "for",
        "of", "on", "at", "be", "do", "i", "you", "we", "this", "that",
        "with", "from", "not", "are", "was", "has", "have", "will", "can",
        "use", "as", "by", "my", "me", "if", "so", "but",
    }
    q_words = {w for w in query.lower().split() if len(w) > 2 and w not in STOPWORDS}
    c_words = {w for w in content.lower().split() if len(w) > 2 and w not in STOPWORDS}
    if not q_words or not c_words:
        return 0.0
    intersection = q_words & c_words
    union = q_words | c_words
    return len(intersection) / len(union)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_relevant(
    message: str,
    memories: list[dict],
    use_embeddings: bool,
    get_embedding_fn,
) -> list[dict]:
    """Score all memories against the query, return top-k above threshold.

    Tries semantic (cosine) scoring first; falls back to keyword Jaccard.
    Each result gets 'final_score' = similarity * temporal_score.
    """
    query_emb: Optional[list[float]] = None
    if use_embeddings:
        try:
            query_emb = get_embedding_fn(message)
        except Exception:
            query_emb = None

    scored = []
    for mem in memories:
        if mem.get("memory_type") not in RETRIEVABLE_TYPES:
            continue

        t_score = _temporal_score(mem)
        if t_score < MIN_TEMPORAL_SCORE:
            continue

        if query_emb is not None and mem.get("embedding") is not None:
            # Semantic path
            from embeddings import cosine_similarity
            sim = cosine_similarity(query_emb, mem["embedding"])
            threshold = SIMILARITY_THRESHOLD
        else:
            # Keyword fallback
            sim = _keyword_similarity(message, mem.get("content", ""))
            threshold = KEYWORD_THRESHOLD

        if sim < threshold:
            continue

        scored.append({
            **mem,
            "similarity": sim,
            "temporal_score": t_score,
            "final_score": sim * (0.5 + t_score * 0.5),  # blend: sim dominant
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:TOP_K]


# ── Context formatting ────────────────────────────────────────────────────────

def format_context(hits: list[dict]) -> str:
    """Format retrieved memories into the additionalContext string."""
    lines = [
        "## Relevant Memories (retrieved for this query)",
        "",
        "The following memories are relevant to what you're about to work on.",
        "Follow any corrections or rules listed here.",
        "",
    ]

    for hit in hits:
        label = TYPE_LABEL.get(hit.get("memory_type", "semantic"), "memory")
        content = hit.get("content", "").strip()
        lines.append(f"- **[{label}]** {content}")

    lines.append("")
    return "\n".join(lines)


# ── Access tracking ───────────────────────────────────────────────────────────

def _record_access(conn, memory_ids: list[str]) -> None:
    """Bump access_count and last_accessed_at for retrieved memories."""
    if not memory_ids:
        return
    now = time.time()
    conn.executemany(
        """
        UPDATE memories
        SET access_count = access_count + 1, last_accessed_at = ?
        WHERE id = ?
        """,
        [(now, mid) for mid in memory_ids],
    )
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Parse args ────────────────────────────────────────────────────────────
    if len(sys.argv) >= 2:
        message = sys.argv[1]
    else:
        message = sys.stdin.read().strip()

    session_id = sys.argv[2] if len(sys.argv) >= 3 else ""
    project = os.environ.get("ENSEMBLE_MEMORY_PROJECT", "") or os.getcwd()

    if not message or len(message) < 5:
        print("{}")
        return

    # ── Import db ─────────────────────────────────────────────────────────────
    try:
        import db
    except ImportError:
        print("{}")
        return

    # ── Connect and ensure embedding column exists ────────────────────────────
    try:
        conn = db.get_db()
    except Exception:
        print("{}")
        return

    _ensure_embedding_column(conn)

    # ── Try to load embeddings module ─────────────────────────────────────────
    use_embeddings = False
    get_embedding_fn = None
    try:
        import embeddings as _emb
        if _emb._AVAILABLE:
            use_embeddings = True
            get_embedding_fn = _emb.get_embedding
            # Backfill any memories that don't have embeddings yet (fast — skips if done)
            _backfill_embeddings(conn, get_embedding_fn)
    except Exception:
        use_embeddings = False

    # ── Load memories (cached after first call) ───────────────────────────────
    try:
        memories = _load_memories(conn, project)
    except Exception:
        conn.close()
        print("{}")
        return

    if not memories:
        conn.close()
        print("{}")
        return

    # ── Retrieve relevant memories ────────────────────────────────────────────
    try:
        hits = retrieve_relevant(message, memories, use_embeddings, get_embedding_fn)
    except Exception:
        conn.close()
        print("{}")
        return

    if not hits:
        conn.close()
        print("{}")
        return

    # ── Record access for retrieved memories ──────────────────────────────────
    try:
        _record_access(conn, [h["id"] for h in hits])
        # Invalidate in-memory cache so access counts are fresh next load
        global _memories_cache
        _memories_cache = None
    except Exception:
        pass

    conn.close()

    # ── Format and output ─────────────────────────────────────────────────────
    context = format_context(hits)
    result = {"additionalContext": context}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
