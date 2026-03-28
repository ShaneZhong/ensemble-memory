#!/usr/bin/env python3
"""store_memory.py — Write extracted memories to SQLite + daily markdown log.

Usage: store_memory.py <extraction_json> <session_id>

Replaces the direct call to write_log.py in stop.sh.  SQLite errors are
non-fatal: if the DB write fails, markdown is still written (markdown is
the source of truth).

Env vars (passed through to write_log):
    ENSEMBLE_MEMORY_LOGS   Daily .md output dir  (default: ~/.ensemble_memory/memory/)
    ENSEMBLE_MEMORY_DIR    User data root         (default: ~/.ensemble_memory/)
    TRANSCRIPT_PATH        Source transcript path  (metadata only)
"""

import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger("ensemble_memory.store_memory")

# ── Locate siblings without hardcoded paths ───────────────────────────────────
_HOOKS_DIR = Path(__file__).parent
sys.path.insert(0, str(_HOOKS_DIR))

import db
import enrich
import kg
import write_log

# Promotion threshold: if a procedural memory is reinforced this many times
# within the window, log it as a candidate for CLAUDE.md promotion.
PROMOTION_THRESHOLD = 5
PROJECT = os.environ.get("ENSEMBLE_MEMORY_PROJECT", os.getcwd())
DAEMON_PORT = int(os.environ.get("ENSEMBLE_MEMORY_DAEMON_PORT", "9876"))


def _notify_daemon_invalidate() -> None:
    """Fire-and-forget POST /invalidate_cache so daemon reloads memories."""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{DAEMON_PORT}/invalidate_cache",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=0.2)
    except Exception:
        pass  # Daemon may not be running — that's fine


def _store_to_sqlite(memories: list[dict], session_id: str, entities_raw: list[dict] = None) -> tuple[int, list[str]]:
    """Insert memories into SQLite, detect supersession and reinforcement.

    Returns (new_count, superseded_ids).
    Logs promotion candidates to stderr if threshold is reached.
    """
    new_count = 0
    superseded_ids: list[str] = []

    for mem in memories:
        mem_type = mem.get("type", "")
        trigger = mem.get("trigger_condition", "")

        # ── Reinforcement check (procedural only) ─────────────────────────────
        existing_id = None
        if mem_type == "procedural" and trigger:
            count = db.get_reinforcement_count(trigger)
            if count > 0:
                # Reinforcement exists; skip inserting a duplicate
                if count >= PROMOTION_THRESHOLD:
                    print(
                        f"[store_memory] PROMOTION CANDIDATE: procedural memory "
                        f"'{trigger[:60]}' reinforced {count}x",
                        file=sys.stderr,
                    )
                continue  # don't insert a new row

        # ── Insert new memory ─────────────────────────────────────────────────
        content = mem.get("content", "")
        importance = mem.get("importance", 5)
        subject = mem.get("subject", "")
        mem_id = db.insert_memory(mem, session_id, PROJECT)
        new_count += 1

        # ── Generate and store embedding via daemon (/embed endpoint) ────────
        embedding = None
        try:
            payload = json.dumps({"text": content}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{DAEMON_PORT}/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                result = json.loads(resp.read())
                embedding = result.get("embedding")
            if embedding:
                db.store_embedding(mem_id, embedding)
        except Exception:
            pass  # Embeddings are best-effort

        # ── Contextual enrichment (non-fatal, best-effort) ───────────────
        if enrich.ENRICHMENT_ENABLED and importance >= enrich.MIN_ENRICHMENT_IMPORTANCE:
            try:
                entity_names = [
                    e.get("name", "") for e in (entities_raw or [])
                    if e.get("name", "")
                ]
                enriched = enrich.enrich_memory(
                    content, mem_type, importance, entity_names, subject,
                )
                if enriched:
                    db.store_enrichment(mem_id, enriched["text"], enriched["quality"])
                    logger.info(
                        "[store_memory] Enriched %s (quality=%.2f)",
                        mem_id[:8],
                        enriched["quality"],
                    )
                    # Re-embed with enriched text
                    try:
                        epayload = json.dumps({"text": enriched["text"][:512]}).encode()
                        ereq = urllib.request.Request(
                            f"http://127.0.0.1:{DAEMON_PORT}/embed",
                            data=epayload,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(ereq, timeout=1.0) as eresp:
                            eresult = json.loads(eresp.read())
                            new_emb = eresult.get("embedding")
                        if new_emb:
                            db.store_embedding(mem_id, new_emb)
                    except Exception:
                        pass  # Re-embed is best-effort
            except Exception as exc:
                logger.warning("[store_memory] Enrichment failed: %s", exc)

        # ── Supersession check (structured: subject+predicate match) ─────────
        predicate = mem.get("predicate", "")
        if subject and predicate:
            superseded = db.detect_supersession(mem_id, subject, predicate)
            if superseded:
                superseded_ids.append(superseded)

        # ── Content-similarity supersession (fallback when no structured triple) ─
        if not subject or not predicate:
            superseded = db.detect_content_supersession(
                mem_id, content, mem_type, threshold=0.6, new_embedding=embedding
            )
            if superseded:
                superseded_ids.append(superseded)

        # ── Decision vault write (if decision_type present) ──────────────────
        # Fallback: if LLM put a decision type in "type" field instead of
        # "decision_type", use it (qwen2.5:3b conflates the two fields)
        _DECISION_TYPES = db._VALID_DECISION_TYPES
        decision_type = mem.get("decision_type")
        if not decision_type and mem_type.upper() in _DECISION_TYPES:
            decision_type = mem_type.upper()
        if decision_type:
            db.insert_decision(
                memory_id=mem_id,
                decision_type=decision_type,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                keywords=mem.get("keywords"),
                files_referenced=mem.get("files_referenced"),
                project=PROJECT,
                session_id=session_id,
            )

    return new_count, superseded_ids


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("Usage: store_memory.py <extraction_json> <session_id>")

    extraction_raw = sys.argv[1]
    session_id = sys.argv[2]

    # Parse extraction JSON early so we can report a clean error if malformed.
    try:
        extraction = json.loads(extraction_raw)
    except json.JSONDecodeError as exc:
        print(f"[store_memory] invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    memories = extraction.get("memories", [])

    # Extract entities early so enrichment can use them during SQLite write
    entities_raw = extraction.get("entities", [])

    # ── SQLite write (non-fatal) ──────────────────────────────────────────────
    new_count = 0
    superseded_ids: list[str] = []
    db_ok = True
    try:
        new_count, superseded_ids = _store_to_sqlite(memories, session_id, entities_raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[store_memory] SQLite error (continuing): {exc}", file=sys.stderr)
        db_ok = False

    # ── Debug summary ─────────────────────────────────────────────────────────
    if db_ok:
        parts = [f"{new_count} new"]
        if superseded_ids:
            parts.append(f"{len(superseded_ids)} superseded " + " ".join(superseded_ids))
        print(
            f"[store_memory] Stored {len(memories)} memories ({', '.join(parts)})",
            file=sys.stderr,
        )
        # Notify daemon so it reloads its memory cache on next /search
        _notify_daemon_invalidate()

    # ── KG entity + relationship processing ───────────────────────────────────
    relationships_raw = extraction.get("relationships", [])

    if entities_raw or relationships_raw:
        try:
            # Record episode for this extraction
            episode_id = kg.record_episode(
                session_id=session_id,
                content=extraction_raw[:500],  # truncate for storage
                summary="; ".join(extraction.get("summary", [])),
                entity_names=[e.get("name", "") for e in entities_raw],
            )

            # Upsert entities
            entity_ids = {}
            for ent in entities_raw:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                eid = kg.upsert_entity(
                    name=name,
                    entity_type=ent.get("type", "CONCEPT"),
                    description=ent.get("description"),
                    session_id=session_id,
                )
                entity_ids[name] = eid

            # Insert relationships
            for rel in relationships_raw:
                kg.insert_relationship(
                    subject_name=rel.get("subject", ""),
                    predicate=rel.get("predicate", "RELATED_TO"),
                    object_name=rel.get("object", ""),
                    evidence=rel.get("evidence"),
                    confidence=float(rel.get("confidence", 0.5)),
                    episode_id=episode_id,
                )

            print(
                f"[store_memory] KG: {len(entity_ids)} entities, "
                f"{len(relationships_raw)} relationships",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[store_memory] KG error (continuing): {exc}", file=sys.stderr)

    # ── Markdown write (always runs) ──────────────────────────────────────────
    # Delegate to write_log.main() by temporarily patching sys.argv so it sees
    # the same arguments it expects.
    _orig_argv = sys.argv[:]
    sys.argv = [str(_HOOKS_DIR / "write_log.py"), extraction_raw, session_id]
    try:
        write_log.main()
    finally:
        sys.argv = _orig_argv


if __name__ == "__main__":
    main()
