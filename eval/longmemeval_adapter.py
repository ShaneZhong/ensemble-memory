#!/usr/bin/env python3
"""longmemeval_adapter.py — Ingest LongMemEval conversations into our ensemble memory system.

Parses LongMemEval JSON, converts conversation turns into episodic memories,
batch-embeds them with BGE-M3, and stores in an isolated SQLite DB.

Usage:
    python longmemeval_adapter.py <dataset_path> [--db-path /tmp/longmemeval_eval.db]
"""

import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("longmemeval.adapter")

# Make hooks importable
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
_HOOKS_DIR = _PROJECT_ROOT / "hooks"
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))


DEFAULT_DB_PATH = "/tmp/longmemeval_eval.db"


def parse_timestamp(date_str: str) -> float:
    """Parse LongMemEval date string to Unix timestamp.

    Formats:
        '2023/01/15 10:00'
        '2023/01/15 (Mon) 10:00'
    """
    # Strip day-of-week if present
    if "(" in date_str:
        parts = date_str.split(")")
        date_str = parts[0].split("(")[0].strip() + parts[1].strip()

    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).timestamp()
        except ValueError:
            continue
    # Fallback: return a default
    return datetime(2023, 1, 1).timestamp()


def ingest_question(question_data: dict, db_path: str = DEFAULT_DB_PATH) -> dict:
    """Ingest all sessions for a single LongMemEval question into the eval DB.

    Returns dict with ingestion stats:
        {memories_created: int, sessions_processed: int, question_id: str}
    """
    import db

    # Redirect to eval DB
    db._DB_PATH_OVERRIDE = db_path
    try:
        conn = db.get_db()
        conn.close()
    except Exception:
        pass

    question_id = question_data["question_id"]
    sessions = question_data.get("haystack_sessions", [])
    session_ids = question_data.get("haystack_session_ids", [])
    session_dates = question_data.get("haystack_dates", [])
    answer_session_ids = set(question_data.get("answer_session_ids", []))

    memories_created = 0

    for idx, session_turns in enumerate(sessions):
        session_id = session_ids[idx] if idx < len(session_ids) else f"session_{idx}"
        session_date = session_dates[idx] if idx < len(session_dates) else None
        base_ts = parse_timestamp(session_date) if session_date else time.time()
        is_answer_session = session_id in answer_session_ids

        for turn_idx, turn in enumerate(session_turns):
            content = turn.get("content", "").strip()
            if not content:
                continue

            role = turn.get("role", "user")
            has_answer = turn.get("has_answer", False)

            # Assign importance: answer turns get 8, answer sessions get 6, others 5
            importance = 5
            if has_answer:
                importance = 8
            elif is_answer_session:
                importance = 6

            # Create memory dict matching our insert_memory format
            mem = {
                "content": f"[{role}] {content}",
                "type": "episodic",
                "importance": importance,
                "subject": session_id,
                "predicate": "SAID",
                "object": content[:100],
            }

            # Timestamp: offset each turn slightly for ordering
            created_at = base_ts + (turn_idx * 60)  # 1 minute apart

            content_hash = hashlib.sha256(
                f"{session_id}:{turn_idx}:{content}".encode()
            ).hexdigest()

            # Direct insert (bypass reinforcement/supersession for speed)
            mem_id = str(uuid.uuid4())
            try:
                conn = db.get_db()
                conn.execute(
                    """INSERT OR IGNORE INTO memories
                       (id, content, content_hash, memory_type, importance,
                        extraction_confidence, confidence, subject, predicate, object,
                        session_id, source_expert, project, created_at, decay_rate,
                        stability, access_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        mem_id,
                        mem["content"],
                        content_hash,
                        "episodic",
                        importance,
                        0.9,  # extraction_confidence
                        1.0,  # confidence
                        session_id,
                        "SAID",
                        content[:100],
                        session_id,
                        "longmemeval",
                        "longmemeval",
                        created_at,
                        0.16,  # episodic decay rate
                        max(0.0, min(1.0, (importance - 1) / 9.0)),
                        0,
                    ),
                )
                conn.commit()
                conn.close()
                memories_created += 1
            except Exception as exc:
                logger.warning("Insert failed for %s turn %d: %s", session_id, turn_idx, exc)

    return {
        "question_id": question_id,
        "memories_created": memories_created,
        "sessions_processed": len(sessions),
    }


def batch_embed_memories(db_path: str = DEFAULT_DB_PATH, batch_size: int = 64) -> int:
    """Batch-embed all memories in the eval DB that lack embeddings.

    Returns count of newly embedded memories.
    """
    import db
    import embeddings

    db._DB_PATH_OVERRIDE = db_path

    # Ensure embedding column exists
    try:
        from db_memory import ensure_embedding_column
        ensure_embedding_column()
    except Exception:
        pass

    conn = db.get_db()
    try:
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE embedding IS NULL"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0

    embedded = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [r["content"] for r in batch]
        ids = [r["id"] for r in batch]

        try:
            embs = embeddings.get_embeddings(texts)
        except Exception as exc:
            logger.warning("Batch embedding failed at offset %d: %s", i, exc)
            continue

        conn = db.get_db()
        try:
            for mid, emb in zip(ids, embs):
                conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (json.dumps(emb), mid),
                )
            conn.commit()
            embedded += len(batch)
        finally:
            conn.close()

        if (i + batch_size) % 256 == 0:
            logger.info("Embedded %d / %d memories", embedded, len(rows))

    return embedded


def ingest_dataset(
    dataset_path: str,
    db_path: str = DEFAULT_DB_PATH,
    limit: int | None = None,
) -> dict:
    """Ingest full LongMemEval dataset. Returns stats dict."""
    import db

    # Clean start: remove existing eval DB
    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize schema
    db._DB_PATH_OVERRIDE = db_path
    conn = db.get_db()
    conn.close()

    # Ensure embedding column
    try:
        from db_memory import ensure_embedding_column, ensure_enrichment_columns
        ensure_embedding_column()
        ensure_enrichment_columns()
    except Exception:
        pass

    with open(dataset_path) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    total_memories = 0
    total_sessions = 0
    seen_sessions = set()

    for i, question_data in enumerate(dataset):
        # Only ingest sessions we haven't seen yet
        new_sessions = []
        new_session_indices = []
        for idx, sid in enumerate(question_data.get("haystack_session_ids", [])):
            if sid not in seen_sessions:
                seen_sessions.add(sid)
                new_sessions.append(sid)
                new_session_indices.append(idx)

        if not new_session_indices:
            continue

        # Build a filtered question_data with only new sessions
        filtered = {
            **question_data,
            "haystack_sessions": [
                question_data["haystack_sessions"][idx]
                for idx in new_session_indices
                if idx < len(question_data["haystack_sessions"])
            ],
            "haystack_session_ids": [
                question_data["haystack_session_ids"][idx]
                for idx in new_session_indices
                if idx < len(question_data["haystack_session_ids"])
            ],
            "haystack_dates": [
                question_data["haystack_dates"][idx]
                for idx in new_session_indices
                if idx < len(question_data["haystack_dates"])
            ],
        }

        stats = ingest_question(filtered, db_path)
        total_memories += stats["memories_created"]
        total_sessions += stats["sessions_processed"]

        if (i + 1) % 50 == 0:
            logger.info(
                "Ingested %d/%d questions (%d memories, %d sessions)",
                i + 1, len(dataset), total_memories, total_sessions,
            )

    return {
        "questions": len(dataset),
        "memories_created": total_memories,
        "sessions_processed": total_sessions,
        "unique_sessions": len(seen_sessions),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest LongMemEval into ensemble memory")
    parser.add_argument("dataset_path", help="Path to longmemeval_oracle.json")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions to ingest")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    args = parser.parse_args()

    print(f"Ingesting {args.dataset_path} → {args.db_path}")
    stats = ingest_dataset(args.dataset_path, args.db_path, args.limit)
    print(f"Ingestion complete: {stats}")

    if not args.skip_embed:
        print("Batch embedding memories...")
        embedded = batch_embed_memories(args.db_path)
        print(f"Embedded {embedded} memories")
