#!/usr/bin/env python3
"""embedding_daemon.py — Persistent HTTP server that keeps the embedding model warm.

Exposes:
    GET  /health            → {"status": "ok"}
    POST /search            → {"query": "...", "project": "..."} → {"hits": [...], "context": "..."}
    POST /invalidate_cache  → {} → {} (bust in-memory memories cache)

The model is loaded once at startup. All subsequent /search calls embed the
query and score memories in-process — no cold-start penalty.

Env vars:
    ENSEMBLE_MEMORY_DAEMON_PORT  Port to listen on (default: 9876)
    ENSEMBLE_MEMORY_DIR          Data dir (default: ~/.ensemble_memory)
    ENSEMBLE_MEMORY_HOME         Project root (to locate hooks/db.py etc.)
"""

import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logger = logging.getLogger("ensemble_memory.daemon")

# ── Locate hooks dir and make siblings importable ─────────────────────────────
_DAEMON_DIR = Path(__file__).resolve().parent
_HOME = Path(os.environ.get("ENSEMBLE_MEMORY_HOME", str(_DAEMON_DIR.parent)))
_HOOKS_DIR = _HOME / "hooks"
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("ENSEMBLE_MEMORY_DAEMON_PORT", "9876"))

SIMILARITY_THRESHOLD = 0.20
KEYWORD_THRESHOLD    = 0.15
TOP_K                = 5
MIN_TEMPORAL_SCORE   = 0.05
MIN_IMPORTANCE       = 3
CACHE_TTL_SECONDS    = 300

RETRIEVABLE_TYPES = {"procedural", "correction", "semantic", "episodic"}
TYPE_LABEL = {
    "correction": "correction",
    "procedural": "procedural",
    "semantic":   "fact",
    "episodic":   "context",
}

# ── Model + cache state (module-level, lives for the daemon's lifetime) ────────
_model = None          # sentence_transformers.SentenceTransformer or None
_has_embeddings = False

_cross_encoder = None  # sentence_transformers.CrossEncoder or None
_has_cross_encoder = False
_cross_encoder_lock = threading.Lock()

_memories_cache: list[dict] = []
_cache_loaded_at: float = 0.0


def _load_model() -> None:
    global _model, _has_embeddings
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _has_embeddings = True
        logger.info("Embedding model loaded")
    except Exception as exc:
        _model = None
        _has_embeddings = False
        logger.warning("sentence-transformers unavailable (%s), using keyword fallback", exc)


def _get_embedding(text: str) -> list[float] | None:
    if _model is None:
        return None
    try:
        vec = _model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception:
        return None


def _load_cross_encoder() -> None:
    """Lazily load the cross-encoder model (thread-safe)."""
    global _cross_encoder, _has_cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return
        if os.environ.get("ENSEMBLE_MEMORY_CROSS_ENCODER", "1") != "1":
            _has_cross_encoder = False
            return
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            _has_cross_encoder = True
            logger.info("Cross-encoder model loaded")
        except Exception as exc:
            _cross_encoder = None
            _has_cross_encoder = False
            logger.warning("Cross-encoder unavailable (%s), reranking disabled", exc)


def _truncate_for_rerank(content: str, subject: str | None) -> str:
    """Truncate long content for cross-encoder input.

    For content >600 chars: use subject as heading + first 200 chars.
    """
    if len(content) <= 600:
        return content
    prefix = f"{subject}: " if subject else ""
    return prefix + content[:200]


def _cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    top_n: int = TOP_K,
) -> list[dict]:
    """Rerank candidates using cross-encoder. Returns top_n results.

    Falls back to returning candidates unchanged if cross-encoder unavailable.
    """
    if not candidates:
        return candidates

    candidates = list(candidates)  # defensive copy — avoid mutating caller's list

    _load_cross_encoder()
    if not _has_cross_encoder or _cross_encoder is None:
        return candidates[:top_n]

    # Build query-document pairs
    pairs = []
    for c in candidates:
        doc = _truncate_for_rerank(
            c.get("content", ""),
            c.get("subject"),
        )
        pairs.append([query, doc])

    try:
        scores = _cross_encoder.predict(pairs)
        for i, c in enumerate(candidates):
            c["cross_encoder_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        reranked = candidates[:top_n]
        # Replace final_score with cross-encoder score (pure reorder)
        for item in reranked:
            item["final_score"] = item["cross_encoder_score"]
        return reranked
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed: %s", exc)
        return candidates[:top_n]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    from embeddings import cosine_similarity
    return cosine_similarity(a, b)


def _keyword_similarity(query: str, content: str) -> float:
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
    return len(q_words & c_words) / len(q_words | c_words)


def _temporal_score(row: dict) -> float:
    """Compute temporal score using canonical formula from db module."""
    import db
    return db.temporal_score(
        access_count=int(row.get("access_count", 0)),
        last_accessed_at=row.get("last_accessed_at"),
        created_at=row.get("created_at", time.time()),
        decay_rate=float(row.get("decay_rate", 0.16)),
        stability=float(row.get("stability", 0.0)),
    )


def _load_memories(project: str) -> list[dict]:
    """Load memories from SQLite. Returns cached list if within TTL."""
    global _memories_cache, _cache_loaded_at

    now = time.time()
    if _memories_cache and (now - _cache_loaded_at) < CACHE_TTL_SECONDS:
        return _memories_cache

    try:
        import db
    except ImportError:
        return []

    try:
        conn = db.get_db()
    except Exception:
        return []

    params: list = [MIN_IMPORTANCE]
    project_clause = ""
    if project:
        project_clause = (
            "AND (project = ? OR project LIKE ? || '/%' OR ? LIKE project || '/%')"
        )
        params.extend([project, project, project])

    try:
        # Phase 6: superseded_by IS NULL AND gc_eligible = 0 filters ensure
        # superseded and garbage-collected memories are excluded from retrieval.
        rows = conn.execute(
            f"""
            SELECT id, content, memory_type, importance, subject,
                   access_count, last_accessed_at, created_at,
                   decay_rate, stability, embedding,
                   temporal_score, score_computed_at
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
    except Exception:
        conn.close()
        return []

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

    conn.close()
    _memories_cache = result
    _cache_loaded_at = now
    return result


def _record_access(memory_ids: list[str]) -> None:
    if not memory_ids:
        return
    try:
        import db
        conn = db.get_db()
        now = time.time()
        conn.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


_KG_STOP_WORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "for", "of", "to", "in", "on", "at", "by", "with", "from", "up",
    "about", "into", "through", "during", "before", "after",
    "use", "using", "used", "get", "set", "make", "let", "just",
])


def _get_kg_context(query_text: str) -> str:
    """Get KG neighborhood context for entities mentioned in query.

    Extracts keywords from the query and searches FTS5 per-keyword,
    since FTS5 requires exact token matches (no stemming).
    """
    try:
        hooks_dir = str(Path(__file__).parent.parent / "hooks")
        if hooks_dir not in sys.path:
            sys.path.insert(0, hooks_dir)
        import kg

        # Extract meaningful keywords from query
        import re
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_.-]*', query_text)
        keywords = [w for w in words if w.lower() not in _KG_STOP_WORDS and len(w) > 2]

        # Generate stem-like variants by stripping common suffixes
        expanded = set()
        for kw in keywords:
            expanded.add(kw)
            low = kw.lower()
            for suffix in ("ing", "tion", "ed", "er", "est", "ly", "ment", "ness", "es", "s"):
                if low.endswith(suffix) and len(low) - len(suffix) >= 3:
                    expanded.add(kw[:len(kw) - len(suffix)])
                    break

        # Search FTS5 per keyword, collect unique entities
        seen_ids = set()
        entities = []
        for kw in expanded:
            hits = kg.search_entities_fts(kw, limit=3)
            for ent in hits:
                if ent["id"] not in seen_ids:
                    seen_ids.add(ent["id"])
                    entities.append(ent)

        if not entities:
            return ""

        entity_names = [e["name"] for e in entities[:5]]
        neighborhood = kg.kg_entity_neighborhood(entity_names, max_depth=2, max_neighbors=3)

        # Community-aware soft preference: sort neighborhood entities so that
        # entities sharing a community with any query entity come first.
        nb_entities = neighborhood.get("entities", [])
        if nb_entities:
            query_community_ids = set()
            for e in entities[:5]:
                cid = e.get("community_id")
                if cid is not None:
                    query_community_ids.add(cid)

            if query_community_ids:
                nb_entities.sort(
                    key=lambda ent: (
                        0 if ent.get("community_id") in query_community_ids else 1,
                        ent.get("name", ""),
                    )
                )
                neighborhood["entities"] = nb_entities

        return neighborhood.get("formatted_prefix", "")
    except Exception:
        return ""


def _bm25_search(query: str, project: str, limit: int = 20) -> list[dict]:
    """Search memories + decisions via FTS5 BM25 ranking.

    Queries both decisions_fts (Phase 4) and a direct LIKE search on memories
    as BM25 fallback (no FTS5 on memories table yet).
    Returns list of dicts with id, content, bm25_rank.
    """
    try:
        import db
    except ImportError:
        return []

    results = []
    conn = None
    try:
        conn = db.get_db()

        # 1. Search decisions via FTS5 BM25
        try:
            import re
            fts_tokens = re.findall(r'[a-zA-Z0-9_]+', query)
            fts_query = " ".join(f'"{t}"' for t in fts_tokens) if fts_tokens else '""'

            params_d: list = [fts_query]
            project_clause = ""
            if project:
                project_clause = "AND d.project = ?"
                params_d.append(project)
            params_d.append(limit)

            rows = conn.execute(
                f"""
                SELECT d.memory_id AS id, m.content, m.memory_type,
                       m.importance, bm25(decisions_fts) AS bm25_score
                FROM decisions_fts
                JOIN decisions d ON d.rowid = decisions_fts.rowid
                JOIN memories m ON m.id = d.memory_id
                WHERE decisions_fts MATCH ?
                  {project_clause}
                ORDER BY bm25(decisions_fts)
                LIMIT ?
                """,
                params_d,
            ).fetchall()
            for i, row in enumerate(rows):
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "importance": row["importance"],
                    "bm25_rank": i + 1,
                    "source": "decisions_fts",
                })
        except Exception:
            pass  # decisions_fts may not exist yet — graceful degradation

        # 2. Simple keyword search on memories as BM25 proxy
        # Use LIKE with keywords extracted from query
        import re
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_.-]*', query)
        keywords = [w for w in words if w.lower() not in _KG_STOP_WORDS and len(w) > 2]

        if keywords:
            seen_ids = {r["id"] for r in results}
            # Build OR conditions for keyword matching
            like_clauses = " OR ".join(["content LIKE ?" for _ in keywords])
            like_params: list = [f"%{kw}%" for kw in keywords[:5]]  # limit to 5 keywords

            params_m: list = []
            proj_clause_m = ""
            if project:
                proj_clause_m = "AND (project = ? OR project LIKE ? || '/%' OR ? LIKE project || '/%')"
                params_m.extend([project, project, project])

            mem_rows = conn.execute(
                f"""
                SELECT id, content, memory_type, importance
                FROM memories
                WHERE ({like_clauses})
                  AND superseded_by IS NULL
                  AND gc_eligible = 0
                  {proj_clause_m}
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                like_params + params_m + [limit],
            ).fetchall()

            for i, row in enumerate(mem_rows):
                if row["id"] not in seen_ids:
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "memory_type": row["memory_type"],
                        "importance": row["importance"],
                        "bm25_rank": len(results) + 1,
                        "source": "memories_keyword",
                    })

    except Exception as exc:
        logger.error("BM25 search error: %s", exc)
    finally:
        if conn:
            conn.close()

    return results


def composite_score(
    rrf_score: float,
    temporal_score: float,
    importance: int,
    confidence: float = 1.0,
) -> float:
    """Composite scoring: weighted sum of RRF, temporal, and importance.

    Replaces the old multiplicative formula. Key improvement: confidence
    (reduced on contradiction) now affects ranking.

    Args:
        rrf_score: Normalized RRF from fusion
        temporal_score: From db.temporal_score() [0, 1]
        importance: Memory importance [1, 10]
        confidence: Retrieval confidence [0, 1], reduced on contradiction

    Returns:
        Final score in [0, ~1].
    """
    w_semantic = 0.5
    w_temporal = 0.3
    w_importance = 0.2

    importance_score = importance / 10.0

    final = (
        w_semantic * rrf_score
        + w_temporal * temporal_score
        + w_importance * importance_score
    )
    final *= max(0.0, min(1.0, confidence))

    return final


def _search(query: str, project: str, rerank: bool = False) -> dict:
    """Core search logic with RRF fusion + optional cross-encoder reranking.

    Args:
        query: Search query text.
        project: Project path for filtering.
        rerank: If True, apply cross-encoder reranking to top-20 RRF results.
                Default False (UserPromptSubmit). Stop hook callers pass True.
    """
    memories = _load_memories(project)
    if not memories:
        return {"hits": [], "context": ""}

    query_emb = _get_embedding(query) if _has_embeddings else None

    # ── Signal 1: Cosine similarity ranking ─────────────────────────────
    cosine_ranked = []
    now = time.time()
    for mem in memories:
        if mem.get("memory_type") not in RETRIEVABLE_TYPES:
            continue
        # Validity gates: exclude superseded, gc_eligible, or expired memories
        if mem.get("superseded_by") is not None:
            continue
        if mem.get("gc_eligible", 0) == 1:
            continue
        valid_to = mem.get("valid_to")
        if valid_to is not None and valid_to < now:
            continue
        # Use cached temporal_score if computed within 6 hours
        _CACHE_FRESHNESS = 21600  # 6 hours in seconds
        cached_score = mem.get("temporal_score")
        computed_at = mem.get("score_computed_at")
        if (cached_score is not None
                and computed_at is not None
                and (now - computed_at) < _CACHE_FRESHNESS):
            t_score = cached_score
        else:
            t_score = _temporal_score(mem)
        if t_score < MIN_TEMPORAL_SCORE:
            continue

        if query_emb is not None and mem.get("embedding") is not None:
            sim = _cosine_similarity(query_emb, mem["embedding"])
        else:
            sim = _keyword_similarity(query, mem.get("content", ""))

        cosine_ranked.append({
            "id": mem["id"],
            "content": mem.get("content", ""),
            "memory_type": mem.get("memory_type", ""),
            "importance": mem.get("importance", 5),
            "confidence": mem.get("confidence", 1.0),
            "subject": mem.get("subject"),
            "similarity": sim,
            "temporal_score": t_score,
            "source": "cosine",
            **{k: mem[k] for k in ("access_count", "last_accessed_at", "created_at",
                                    "decay_rate", "stability", "superseded_by",
                                    "gc_eligible", "valid_to") if k in mem},
        })

    cosine_ranked.sort(key=lambda x: x["similarity"], reverse=True)

    # ── Signal 2: BM25 ranking ──────────────────────────────────────────
    bm25_results = _bm25_search(query, project, limit=20)

    # ── RRF Fusion ──────────────────────────────────────────────────────
    RRF_K = 60
    rrf_scores: dict[str, float] = {}
    all_items: dict[str, dict] = {}

    # Cosine signal (top 50)
    for rank, item in enumerate(cosine_ranked[:50], start=1):
        mid = item["id"]
        rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)
        all_items[mid] = item

    # BM25 signal
    for bm25_item in bm25_results:
        mid = bm25_item["id"]
        rank = bm25_item["bm25_rank"]
        rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)
        if mid not in all_items:
            all_items[mid] = {
                "id": mid,
                "content": bm25_item.get("content", ""),
                "memory_type": bm25_item.get("memory_type", ""),
                "importance": bm25_item.get("importance", 5),
                "similarity": 0.0,
                "temporal_score": 0.5,  # default for BM25-only hits
            }

    # Build final scored list
    use_composite = os.environ.get("ENSEMBLE_MEMORY_COMPOSITE_SCORING", "1") == "1"
    scored = []
    for mid, rrf_score in rrf_scores.items():
        item = all_items[mid]
        t_score = item.get("temporal_score", 0.5)
        importance_raw = item.get("importance", 5)
        confidence = item.get("confidence", 1.0) or 1.0

        if use_composite:
            final = composite_score(rrf_score, t_score, importance_raw, confidence)
        else:
            # Legacy formula (pre-Phase 7)
            final = rrf_score * (0.4 + t_score * 0.3 + (importance_raw / 10.0) * 0.3)

        scored.append({
            **item,
            "rrf_score": rrf_score,
            "final_score": final,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # ── Optional cross-encoder reranking (Stop hook only) ─────────────
    if rerank and os.environ.get("ENSEMBLE_MEMORY_CROSS_ENCODER", "1") == "1":
        hits = _cross_encoder_rerank(query, scored[:20], top_n=TOP_K)
    else:
        hits = scored[:TOP_K]

    if not hits:
        return {"hits": [], "context": ""}

    _record_access([h["id"] for h in hits])

    context_parts = [_format_context(hits)]
    kg_context = _get_kg_context(query)
    if kg_context:
        context_parts.append(f"\n## Related Knowledge\n{kg_context}")
    context = "".join(context_parts)

    return {"hits": hits, "context": context}


def _format_context(hits: list[dict]) -> str:
    lines = [
        "## Relevant Memories (retrieved for this query)",
        "",
        "The following memories are relevant to what you're about to work on.",
        "Follow any corrections or rules listed here.",
        "",
    ]
    for hit in hits:
        label   = TYPE_LABEL.get(hit.get("memory_type", "semantic"), "memory")
        content = hit.get("content", "").strip()
        lines.append(f"- **[{label}]** {content}")
    lines.append("")
    return "\n".join(lines)


# ── Background jobs ────────────────────────────────────────────────────────────

_bg_timer = None
_daemon_start_time: float = 0.0
_last_bg_job_time: float | None = None


def _process_amem_queue():
    """Process pending A-MEM evolution queue items.

    For each queued memory, find similar memories via embedding search,
    classify relationships via Ollama, and write to kg_memory_links.
    Wrapped in own try/except so failures don't break the timer chain.
    """
    import db

    pending = db.get_pending_amem_queue(max_age_days=7)
    if not pending:
        return

    try:
        import evolution
    except ImportError:
        logger.warning("[daemon] evolution module not available, skipping A-MEM")
        return

    processed = 0
    for queue_key, memory_id in pending:
        try:
            # Get the memory content
            conn = db.get_db()
            try:
                row = conn.execute(
                    "SELECT content FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
            finally:
                conn.close()

            if not row:
                db.dequeue_amem(queue_key)
                continue

            content = row["content"]

            # Find similar memories via embedding search
            similar = _find_similar_for_amem(memory_id, content)
            if not similar:
                db.dequeue_amem(queue_key)
                continue

            # Classify relationships
            relationships = evolution.classify_relationships(
                memory_id, content, similar,
            )

            # Write to amem_memory_links
            for rel in relationships:
                db.insert_memory_link(
                    source_memory_id=rel["source_entity_id"],
                    target_memory_id=rel["target_entity_id"],
                    link_type=rel["link_type"],
                    strength=rel["strength"],
                )
                logger.info(
                    "[daemon] A-MEM link: %s -[%s]-> %s (%.2f)",
                    memory_id[:8], rel["link_type"],
                    rel["target_entity_id"][:8], rel["strength"],
                )

            db.dequeue_amem(queue_key)
            processed += 1

        except Exception as exc:
            # Leave in queue for retry — 7-day expiry will clean up
            logger.warning(
                "[daemon] A-MEM failed for %s: %s", memory_id[:8], exc,
            )

    if processed > 0:
        logger.info("[daemon] A-MEM: processed %d/%d queued memories", processed, len(pending))


def _find_similar_for_amem(memory_id: str, content: str) -> list[dict]:
    """Find top-5 similar memories for A-MEM classification (excluding self)."""
    import db

    # Use the daemon's own embedding infrastructure
    embedding = _get_embedding(content)
    if embedding is None:
        return []

    conn = db.get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, content, embedding FROM memories
            WHERE id != ? AND embedding IS NOT NULL
              AND superseded_by IS NULL AND gc_eligible = 0
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (memory_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    import json as _json
    try:
        from embeddings import cosine_similarity
    except ImportError:
        return []

    scored = []
    for row in rows:
        try:
            cand_emb = _json.loads(row["embedding"])
            sim = cosine_similarity(embedding, cand_emb)
            if sim >= 0.3:  # Low threshold — Ollama will judge relevance
                scored.append({"id": row["id"], "content": row["content"], "score": sim})
        except (ValueError, TypeError):
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]


def _run_background_jobs():
    """Run background maintenance jobs (serialized to prevent SQLite lock contention)."""
    global _bg_timer, _last_bg_job_time
    try:
        import db
        import kg

        # 1. Temporal score batch computation
        count = db.compute_temporal_scores()
        if count > 0:
            logger.info("[daemon] Computed temporal scores for %d memories", count)

        # 2. Process supersession events
        event_stats = db.process_supersession_events()
        if event_stats['processed'] > 0:
            logger.info("[daemon] Processed %d supersession events", event_stats['processed'])

        # 3. Enforce chain depth limits
        chain_marked = db.enforce_chain_depth_limits()
        if chain_marked > 0:
            logger.info("[daemon] Chain pruning marked %d memories as gc_eligible", chain_marked)

        # 4. Garbage collection
        gc_stats = db.run_garbage_collection()
        if gc_stats['total'] > 0:
            logger.info("[daemon] GC: %d forgotten", gc_stats['gc_forgotten'])

        # 5. Promotion scan (check for memories eligible for CLAUDE.md promotion)
        try:
            import promote
            conn = db.get_db()
            try:
                candidates = conn.execute("""
                    SELECT id FROM memories
                    WHERE promotion_candidate = 1
                      AND reinforcement_count >= 5
                      AND memory_type = 'procedural'
                      AND superseded_by IS NULL
                      AND gc_eligible = 0
                """).fetchall()
            finally:
                conn.close()
            for candidate in candidates:
                promote.check_and_promote(candidate["id"])
        except Exception as exc:
            logger.warning("[daemon] Promotion scan failed: %s", exc)

        # 6. Community detection
        try:
            communities = kg.detect_communities()
            if communities > 0:
                logger.info("[daemon] Detected %d communities", communities)
        except Exception as exc:
            logger.warning("[daemon] Community detection failed: %s", exc)

        # 7. Relationship decay
        try:
            decay_stats = kg.apply_relationship_decay()
            if decay_stats['decayed'] > 0 or decay_stats['expired'] > 0:
                logger.info("[daemon] Relationship decay: %d decayed, %d expired",
                           decay_stats['decayed'], decay_stats['expired'])
        except Exception as exc:
            logger.warning("[daemon] Relationship decay failed: %s", exc)

        # 8. A-MEM evolution (process queued memories)
        try:
            _process_amem_queue()
        except Exception as exc:
            logger.warning("[daemon] A-MEM evolution failed: %s", exc)

        _last_bg_job_time = time.time()
    except Exception as exc:
        logger.warning("[daemon] Background job failed: %s", exc)
        _last_bg_job_time = time.time()

    # Schedule next run in 6 hours
    _bg_timer = threading.Timer(21600, _run_background_jobs)
    _bg_timer.daemon = True
    _bg_timer.start()


# ── Status builder ─────────────────────────────────────────────────────────────

def _build_status() -> dict:
    """Build status dict with daemon health metrics."""
    import datetime

    now = time.time()
    status: dict = {
        "status": "ok",
        "port": PORT,
        "model_loaded": _has_embeddings,
        "memory_count": 0,
        "cache_size": len(_memories_cache),
        "cache_age_seconds": round(now - _cache_loaded_at, 1) if _cache_loaded_at else None,
        "last_background_job": None,
        "kg_entities": 0,
        "kg_communities": 0,
        "uptime_seconds": round(now - _daemon_start_time, 1) if _daemon_start_time else 0,
    }

    if _last_bg_job_time is not None:
        status["last_background_job"] = (
            datetime.datetime.fromtimestamp(_last_bg_job_time, tz=datetime.timezone.utc)
            .isoformat()
        )

    try:
        import db
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE superseded_by IS NULL AND gc_eligible = 0"
            ).fetchone()
            status["memory_count"] = row["cnt"] if row else 0

            row = conn.execute("SELECT COUNT(*) AS cnt FROM kg_entities").fetchone()
            status["kg_entities"] = row["cnt"] if row else 0

            row = conn.execute(
                "SELECT COUNT(DISTINCT community_id) AS cnt FROM kg_entities WHERE community_id IS NOT NULL"
            ).fetchone()
            status["kg_communities"] = row["cnt"] if row else 0
        finally:
            conn.close()
    except Exception:
        pass

    return status


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        pass

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/status":
            self._send_json(200, _build_status())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/search":
            body    = self._read_body()
            query   = body.get("query", "")
            project = body.get("project", "")
            rerank  = bool(body.get("rerank", False))
            if not query:
                self._send_json(400, {"error": "query required"})
                return
            result = _search(query, project, rerank=rerank)
            self._send_json(200, result)

        elif self.path == "/invalidate_cache":
            global _memories_cache, _cache_loaded_at
            _memories_cache = []
            _cache_loaded_at = 0.0
            self._send_json(200, {"invalidated": True})

        elif self.path == "/embed":
            body = self._read_body()
            text = body.get("text", "")
            if not text:
                self._send_json(400, {"error": "text required"})
                return
            # all-MiniLM-L6-v2 truncates at 256 tokens (~512 chars).
            # Pre-truncate to avoid wasting time on text that won't be encoded.
            text = text[:512]
            embedding = _get_embedding(text)
            if embedding is None:
                self._send_json(503, {"error": "embedding model not available"})
                return
            self._send_json(200, {"embedding": embedding})

        elif self.path == "/embed_batch":
            body = self._read_body()
            texts = body.get("texts", [])
            if not isinstance(texts, list) or not texts:
                self._send_json(400, {"error": "texts must be a non-empty list"})
                return
            embeddings = []
            for text in texts:
                emb = _get_embedding(str(text)[:512])
                embeddings.append(emb)
            if any(e is None for e in embeddings):
                self._send_json(503, {"error": "embedding model not available"})
                return
            self._send_json(200, {"embeddings": embeddings})

        elif self.path == "/log_feedback":
            body = self._read_body()
            query = body.get("query", "")
            feedback_type = body.get("type", "miss")  # "miss" = user said "I already told you"
            session_id = body.get("session_id", "")

            # Log to JSONL file
            import datetime
            log_dir = Path(os.environ.get("ENSEMBLE_MEMORY_DIR",
                          str(Path.home() / ".ensemble_memory"))) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "retrieval_feedback.jsonl"
            entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "query": query[:500],
                "type": feedback_type,
                "session_id": session_id,
            }
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
            self._send_json(200, {"logged": True})

        else:
            self._send_json(404, {"error": "not found"})


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global _daemon_start_time
    _daemon_start_time = time.time()
    logger.info("Starting on port %d", PORT)
    _load_model()

    # Run importance decay on startup (cheap, ~1ms)
    try:
        import db
        decayed = db.decay_stale_importance(days_threshold=30, floor=3)
        if decayed:
            logger.info("Decayed importance for %d stale memories", decayed)
    except Exception:
        pass

    # Start background temporal score computation
    _run_background_jobs()

    server = HTTPServer(("127.0.0.1", PORT), _Handler)
    logger.info("Ready")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    logger.info("Shutting down")


if __name__ == "__main__":
    main()
