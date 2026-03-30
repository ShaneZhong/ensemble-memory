#!/usr/bin/env python3
"""db_lifecycle.py — Reinforcement, supersession event processing, chain management, and GC.

Split from db.py for maintainability. All functions import get_db from db.
"""

import json
import logging
import math
import sqlite3
import time
from typing import Optional

logger = logging.getLogger("ensemble_memory.db")


def get_reinforcement_count(trigger_condition: str) -> int:
    """Count non-superseded memories whose content contains trigger_condition.

    Used for cross-session pattern detection. Returns 0 if trigger_condition
    is empty or no matches found.

    DEPRECATED: Use get_reinforcement_match() instead.
    """
    if not trigger_condition:
        return 0

    from db import get_db
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
    1. Exact triple match (same subject + predicate + object) -- most reliable
    2. Embedding cosine similarity >= 0.90 -- catches rephrased versions
    3. Exact LIKE substring match on content -- final fallback

    Returns (reinforcement_count + 1, memory_id), or (0, None) if no match.
    """
    if not match_text and not subject:
        return 0, None

    from db import get_db
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
    from db_memory import _compute_stability
    from db import get_db
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
    from db import get_db
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
    import db as _db_module
    conn = _db_module.get_db()
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
                    # Call via db module so tests can patch db._process_supersession_kg
                    _db_module._process_supersession_kg(conn, old_id, new_id)
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
                    # Call via db module so tests can patch db._process_supersession_contextual
                    _db_module._process_supersession_contextual(conn, old_id)
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
    from db import get_db
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
    from db import get_db
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
    Does NOT delete rows -- sets gc_eligible = 1 for exclusion from retrieval.

    Returns {'gc_chain_pruned': N, 'gc_forgotten': N, 'total': N}
    All counts are delta (newly GC'd this run), not cumulative.
    """
    from db import get_db
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


def compute_temporal_scores(chunk_size: int = 100) -> int:
    """Batch-compute and cache temporal_score for all active memories.

    Processes in chunks of chunk_size to keep write locks short.
    Updates score_computed_at to prevent redundant computation.

    Returns the number of memories updated.
    """
    from db_memory import temporal_score
    from db import get_db
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
