#!/usr/bin/env python3
"""kg.py — Knowledge graph module for ensemble memory system.

Manages entity resolution, relationship edges, and graph traversal.
All data stored in the shared SQLite database (same as db.py).
"""

import json
import logging
import os
import sys
import time
import uuid
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HOOKS_DIR = Path(__file__).parent
sys.path.insert(0, str(_HOOKS_DIR))
import db

DAEMON_PORT = int(os.environ.get("ENSEMBLE_MEMORY_DAEMON_PORT", "9876"))

# Valid entity types and relationship predicates
ENTITY_TYPES = frozenset([
    'TECHNOLOGY', 'PROJECT', 'TOOL', 'FILE', 'API', 'PERSON',
    'PREFERENCE', 'ERROR', 'CONCEPT', 'ORGANIZATION', 'RULE', 'DECISION'
])

PREDICATES = frozenset([
    'USES', 'DEPENDS_ON', 'CONFLICTS_WITH', 'HAS_CONSTRAINT',
    'HAS_VERSION', 'RUNS_ON', 'APPLIES_TO', 'PREVENTS', 'SUPERSEDES',
    'CAUSED_BY', 'IMPLEMENTED_IN', 'STORED_IN', 'PART_OF', 'AFFECTS',
    'RELATED_TO', 'WORKS_ON'
])


def _get_embedding_via_daemon(text: str) -> Optional[list]:
    """Get embedding from the daemon /embed endpoint. Returns None on failure."""
    try:
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{DAEMON_PORT}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            result = json.loads(resp.read())
            return result.get("embedding")
    except Exception:
        return None


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rebuild_fts5_index():
    """Rebuild the FTS5 index from the kg_entities table.

    Called once on module load to fix any stale/missing FTS5 entries.
    Safe to call multiple times — it drops and recreates all FTS5 rows.
    """
    conn = db.get_db()
    try:
        conn.execute("INSERT INTO kg_entities_fts(kg_entities_fts) VALUES('rebuild')")
        conn.commit()
    except Exception:
        pass
    conn.close()


# Rebuild FTS5 on module load to fix any stale entries
rebuild_fts5_index()


def upsert_entity(
    name: str,
    entity_type: str,
    description: Optional[str] = None,
    aliases: Optional[list] = None,
    temporal_memory_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Insert or merge an entity, returning its id.

    Uses FTS5 to find candidates by name, then cosine similarity of descriptions
    (if both have descriptions and daemon is available) to detect duplicates.
    If cosine > 0.85, merges with the existing entity.
    """
    name = name.strip()
    if not name:
        raise ValueError("entity name must not be empty")

    conn = db.get_db()
    now = time.time()

    # ── Exact name match first (most reliable, FTS5 can be stale) ───────────
    exact_match = conn.execute(
        'SELECT rowid, id, name, aliases, description, session_count '
        'FROM kg_entities WHERE LOWER(name) = LOWER(?)',
        (name,),
    ).fetchone()

    best_match_id = None

    if exact_match:
        best_match_id = exact_match["id"]
    else:
        # ── FTS5 fuzzy candidate lookup (aliases, description) ───────────
        candidates = []
        try:
            fts_query = name.replace('"', '""')
            candidates = conn.execute(
                'SELECT e.rowid, e.id, e.name, e.aliases, e.description, e.session_count '
                'FROM kg_entities e '
                'JOIN kg_entities_fts fts ON e.rowid = fts.rowid '
                'WHERE kg_entities_fts MATCH ?',
                (f'"{fts_query}"',),
            ).fetchall()
        except Exception:
            pass

        # Description-based cosine similarity check on FTS5 candidates
        for cand in candidates:
            if description and cand["description"]:
                emb_new = _get_embedding_via_daemon(description)
                emb_old = _get_embedding_via_daemon(cand["description"])
                if emb_new and emb_old:
                    sim = _cosine_similarity(emb_new, emb_old)
                    if sim > 0.85:
                        best_match_id = cand["id"]
                        break

    if best_match_id:
        # ── Merge: update existing entity ────────────────────────────────────
        existing = conn.execute(
            'SELECT description, aliases, session_count FROM kg_entities WHERE id = ?',
            (best_match_id,),
        ).fetchone()

        # Update description if new one is longer/better
        existing_desc = existing["description"] or ""
        new_desc = description or existing_desc
        if description and len(description) > len(existing_desc):
            new_desc = description

        # Merge aliases (JSON array union)
        existing_aliases = json.loads(existing["aliases"] or "[]")
        new_aliases = aliases or []
        merged_aliases = list(set(existing_aliases) | set(new_aliases))

        # Increment session_count only if this appears to be a different session.
        # We use a 60-second grace window: if the entity was last updated more
        # than 60 seconds ago, treat this as a new session interaction.
        session_count = existing["session_count"] or 0
        existing_row_full = conn.execute(
            "SELECT last_updated FROM kg_entities WHERE id = ?", (best_match_id,)
        ).fetchone()
        last_updated = existing_row_full["last_updated"] if existing_row_full else 0
        if session_id and (now - (last_updated or 0)) > 60:
            session_count += 1

        conn.execute(
            """
            UPDATE kg_entities
            SET description = ?, aliases = ?, last_updated = ?, session_count = ?
            WHERE id = ?
            """,
            (new_desc, json.dumps(merged_aliases), now, session_count, best_match_id),
        )
        # Incremental FTS5 update: delete old entry and re-insert with current values
        try:
            rowid_row = conn.execute(
                "SELECT rowid FROM kg_entities WHERE id = ?", (best_match_id,)
            ).fetchone()
            if rowid_row:
                rowid = rowid_row[0]
                name_row = conn.execute(
                    "SELECT name, aliases, description FROM kg_entities WHERE id = ?",
                    (best_match_id,),
                ).fetchone()
                conn.execute(
                    "DELETE FROM kg_entities_fts WHERE rowid = ?", (rowid,)
                )
                conn.execute(
                    "INSERT INTO kg_entities_fts(rowid, name, aliases, description) VALUES (?, ?, ?, ?)",
                    (rowid, name_row["name"], name_row["aliases"], name_row["description"]),
                )
        except Exception:
            pass
        conn.commit()
        conn.close()
        return best_match_id

    # ── Insert new entity ──────────────────────────────────────────────────────
    entity_id = str(uuid.uuid4())
    aliases_json = json.dumps(aliases or [])
    conn.execute(
        """
        INSERT INTO kg_entities (
            id, name, type, description, aliases,
            first_seen, last_updated, session_count, weight, temporal_memory_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entity_id, name, entity_type, description, aliases_json,
            now, now, 1 if session_id else 0, 0.5, temporal_memory_id,
        ),
    )
    # Incremental FTS5 insert
    try:
        rowid_row = conn.execute(
            "SELECT rowid FROM kg_entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if rowid_row:
            conn.execute(
                "INSERT INTO kg_entities_fts(rowid, name, aliases, description) VALUES (?, ?, ?, ?)",
                (rowid_row[0], name, aliases_json, description),
            )
    except Exception:
        pass
    conn.commit()
    conn.close()
    return entity_id


def insert_relationship(
    subject_name: str,
    predicate: str,
    object_name: str,
    evidence: Optional[str] = None,
    confidence: float = 0.5,
    episode_id: Optional[str] = None,
    temporal_memory_id: Optional[str] = None,
) -> Optional[str]:
    """Insert a relationship between two entities.

    Validates predicate. Creates entities if they don't exist.
    Returns relationship id, or None if predicate is invalid.
    """
    subject_name = subject_name.strip()
    object_name = object_name.strip()
    if not subject_name or not object_name:
        return None

    # Normalize predicate: LLMs sometimes return "CAUSES | AFFECTS" or "USES / DEPENDS_ON".
    # Split on '|' and '/', strip each token, use first valid one.
    if predicate not in PREDICATES:
        normalized = None
        for _part in predicate.replace("/", "|").split("|"):
            _candidate = _part.strip().upper()
            if _candidate in PREDICATES:
                normalized = _candidate
                break
        if normalized:
            predicate = normalized
        else:
            print(
                f"[kg] Invalid predicate '{predicate}', falling back to RELATED_TO",
                file=sys.stderr,
            )
            predicate = "RELATED_TO"

    # Auto-create entities as CONCEPT if they don't exist yet.
    # The correct type will be set when upsert_entity() is called with the
    # full entity data from the extraction pass — this is intentional.
    subject_id = upsert_entity(subject_name, "CONCEPT")
    object_id = upsert_entity(object_name, "CONCEPT")

    conn = db.get_db()
    now = time.time()

    # Check for existing identical relationship
    existing = conn.execute(
        """
        SELECT id, confidence FROM kg_relationships
        WHERE subject_id = ? AND predicate = ? AND object_id = ?
          AND valid_until IS NULL
        """,
        (subject_id, predicate, object_id),
    ).fetchone()

    if existing:
        rel_id = existing["id"]
        # Update confidence if new is higher
        if confidence > existing["confidence"]:
            conn.execute(
                "UPDATE kg_relationships SET confidence = ? WHERE id = ?",
                (confidence, rel_id),
            )
            conn.commit()
        conn.close()
        return rel_id

    rel_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO kg_relationships (
            id, subject_id, predicate, object_id,
            evidence, confidence, valid_from, valid_until,
            episode_id, temporal_memory_id, synced_to_temporal, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 0, ?)
        """,
        (
            rel_id, subject_id, predicate, object_id,
            evidence, confidence, now,
            episode_id, temporal_memory_id, now,
        ),
    )
    conn.commit()
    conn.close()
    return rel_id


def kg_entity_neighborhood(
    entity_names: list,
    max_depth: int = 2,
    max_neighbors: int = 3,
) -> dict:
    """Get KG neighborhood context for a list of entity names.

    Uses BFS via recursive CTE with cycle detection.
    Returns dict with entities, relationships, and formatted_prefix.
    """
    if not entity_names:
        return {"entities": [], "relationships": [], "formatted_prefix": ""}

    conn = db.get_db()

    # Resolve entity names to IDs via FTS5 and exact match
    seed_ids = []
    for name in entity_names:
        name = name.strip()
        if not name:
            continue
        found = None
        try:
            fts_query = name.replace('"', '""')
            row = conn.execute(
                'SELECT e.id FROM kg_entities e '
                'JOIN kg_entities_fts fts ON e.rowid = fts.rowid '
                'WHERE kg_entities_fts MATCH ? LIMIT 1',
                (f'"{fts_query}"',),
            ).fetchone()
            if row:
                found = row["id"]
        except Exception:
            pass
        if not found:
            row = conn.execute(
                'SELECT id FROM kg_entities WHERE LOWER(name) = LOWER(?)',
                (name,),
            ).fetchone()
            if row:
                found = row["id"]
        if found and found not in seed_ids:
            seed_ids.append(found)

    if not seed_ids:
        conn.close()
        return {"entities": [], "relationships": [], "formatted_prefix": ""}

    placeholders = ",".join("?" * len(seed_ids))

    # BFS forward (subject → object)
    bfs_sql = f"""
        WITH RECURSIVE bfs(entity_id, depth, path) AS (
            SELECT id, 0, ',' || id || ','
            FROM kg_entities WHERE id IN ({placeholders})
            UNION ALL
            SELECT r.object_id, b.depth + 1,
                   b.path || r.object_id || ','
            FROM kg_relationships r
            JOIN bfs b ON r.subject_id = b.entity_id
            WHERE b.depth < ?
              AND r.valid_until IS NULL
              AND INSTR(b.path, ',' || r.object_id || ',') = 0
        )
        SELECT DISTINCT entity_id FROM bfs
    """
    forward_rows = conn.execute(bfs_sql, seed_ids + [max_depth]).fetchall()
    forward_ids = {r["entity_id"] for r in forward_rows}

    # BFS reverse (object → subject)
    bfs_rev_sql = f"""
        WITH RECURSIVE bfs(entity_id, depth, path) AS (
            SELECT id, 0, ',' || id || ','
            FROM kg_entities WHERE id IN ({placeholders})
            UNION ALL
            SELECT r.subject_id, b.depth + 1,
                   b.path || r.subject_id || ','
            FROM kg_relationships r
            JOIN bfs b ON r.object_id = b.entity_id
            WHERE b.depth < ?
              AND r.valid_until IS NULL
              AND INSTR(b.path, ',' || r.subject_id || ',') = 0
        )
        SELECT DISTINCT entity_id FROM bfs
    """
    reverse_rows = conn.execute(bfs_rev_sql, seed_ids + [max_depth]).fetchall()
    reverse_ids = {r["entity_id"] for r in reverse_rows}

    all_entity_ids = forward_ids | reverse_ids
    if not all_entity_ids:
        conn.close()
        return {"entities": [], "relationships": [], "formatted_prefix": ""}

    # Limit neighbors (keep seeds + up to max_neighbors per seed)
    non_seeds = all_entity_ids - set(seed_ids)
    limited_ids = set(seed_ids) | set(list(non_seeds)[:max_neighbors * len(seed_ids)])

    eid_placeholders = ",".join("?" * len(limited_ids))
    entity_rows = conn.execute(
        f"SELECT id, name, type, description, community_id FROM kg_entities WHERE id IN ({eid_placeholders})",
        list(limited_ids),
    ).fetchall()

    rel_rows = conn.execute(
        f"""
        SELECT r.id, e1.name AS subject_name, r.predicate, e2.name AS object_name,
               r.evidence, r.confidence
        FROM kg_relationships r
        JOIN kg_entities e1 ON r.subject_id = e1.id
        JOIN kg_entities e2 ON r.object_id = e2.id
        WHERE r.subject_id IN ({eid_placeholders})
          AND r.object_id IN ({eid_placeholders})
          AND r.valid_until IS NULL
        """,
        list(limited_ids) + list(limited_ids),
    ).fetchall()

    conn.close()

    entities = [dict(r) for r in entity_rows]
    relationships = [dict(r) for r in rel_rows]

    # Format as natural language prefix
    lines = []
    if entities:
        lines.append("## Knowledge Graph Context")
        lines.append("")
        lines.append("**Entities:**")
        for ent in entities:
            desc = f" — {ent['description']}" if ent.get("description") else ""
            lines.append(f"- {ent['name']} ({ent['type']}){desc}")
        lines.append("")
    if relationships:
        lines.append("**Relationships:**")
        for rel in relationships:
            lines.append(
                f"- {rel['subject_name']} --[{rel['predicate']}]--> {rel['object_name']}"
            )
        lines.append("")

    formatted_prefix = "\n".join(lines) if lines else ""

    return {
        "entities": entities,
        "relationships": relationships,
        "formatted_prefix": formatted_prefix,
    }


def search_entities_fts(query: str, limit: int = 10) -> list:
    """FTS5 search on kg_entities_fts. Returns list of entity dicts."""
    conn = db.get_db()
    results = []

    # Try FTS5 first
    try:
        fts_query = query.replace('"', '""')
        rows = conn.execute(
            """
            SELECT e.id, e.name, e.type, e.description, e.aliases, e.community_id
            FROM kg_entities e
            JOIN kg_entities_fts fts ON e.rowid = fts.rowid
            WHERE kg_entities_fts MATCH ?
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        results = [dict(r) for r in rows]
    except Exception:
        pass

    # Fallback to LIKE if FTS5 returned nothing
    if not results:
        try:
            rows = conn.execute(
                "SELECT id, name, type, description, aliases, community_id FROM kg_entities "
                "WHERE LOWER(name) LIKE LOWER(?) OR LOWER(description) LIKE LOWER(?) "
                "LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            results = [dict(r) for r in rows]
        except Exception:
            pass

    conn.close()
    return results


def get_entity_stats() -> dict:
    """Return counts of KG entities, relationships, episodes, and links."""
    conn = db.get_db()
    try:
        entities = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        relationships = conn.execute(
            "SELECT COUNT(*) FROM kg_relationships WHERE valid_until IS NULL"
        ).fetchone()[0]
        episodes = conn.execute("SELECT COUNT(*) FROM kg_episodes").fetchone()[0]
        links = conn.execute("SELECT COUNT(*) FROM kg_memory_links").fetchone()[0]
    except Exception:
        entities = relationships = episodes = links = 0
    conn.close()
    return {
        "total_entities": entities,
        "total_relationships": relationships,
        "total_episodes": episodes,
        "total_links": links,
    }


def record_episode(
    session_id: str,
    content: str,
    summary: Optional[str] = None,
    entity_names: Optional[list] = None,
) -> str:
    """Insert a kg_episodes row and link entities via kg_appears_in.

    Returns the episode id.
    """
    episode_id = str(uuid.uuid4())
    now = time.time()

    # Insert the episode first and close the connection before calling
    # upsert_entity (which opens its own connection) to avoid SQLite lock.
    conn = db.get_db()
    conn.execute(
        """
        INSERT INTO kg_episodes (id, session_id, session_timestamp, content, summary)
        VALUES (?, ?, ?, ?, ?)
        """,
        (episode_id, session_id, now, content, summary),
    )
    conn.commit()
    conn.close()

    # Upsert each entity (each opens/closes its own connection)
    entity_ids = []
    for name in (entity_names or []):
        name = (name or "").strip()
        if not name:
            continue
        try:
            entity_id = upsert_entity(name, "CONCEPT", session_id=session_id)
            entity_ids.append(entity_id)
        except Exception as exc:
            print(f"[kg] record_episode entity error ({name}): {exc}", file=sys.stderr)

    # Link entities to episode in a fresh connection
    if entity_ids:
        conn = db.get_db()
        for entity_id in entity_ids:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO kg_appears_in (entity_id, episode_id) VALUES (?, ?)",
                    (entity_id, episode_id),
                )
            except Exception as exc:
                print(f"[kg] record_episode link error ({entity_id}): {exc}", file=sys.stderr)
        conn.commit()
        conn.close()

    return episode_id


def bootstrap_from_files(file_paths: list) -> dict:
    """Extract entities and relationships from files using Ollama.

    Reads each file, calls qwen2.5:3b to extract structured KG data,
    then upserts all entities and relationships into the database.
    Returns stats dict.
    """
    import urllib.error

    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    MODEL = os.environ.get("ENSEMBLE_MEMORY_MODEL", "qwen2.5:3b")

    entities_created = 0
    relationships_created = 0

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"[kg] bootstrap: file not found: {file_path}", file=sys.stderr)
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"[kg] bootstrap: error reading {file_path}: {exc}", file=sys.stderr)
            continue

        # Chunk content to stay within model context and timeout limits
        CHUNK_SIZE = 2000
        chunks = []
        for i in range(0, len(content), CHUNK_SIZE):
            chunks.append(content[i:i + CHUNK_SIZE])

        for chunk_idx, chunk in enumerate(chunks):
            prompt = (
                "You are a knowledge graph extraction system. "
                "Read the following file content and extract entities and relationships. "
                "Output ONLY valid JSON.\n\n"
                f"File: {path.name} (part {chunk_idx + 1}/{len(chunks)})\n\n"
                f"Content:\n{chunk}\n\n"
                "Extract entities and relationships matching this schema:\n"
                "{\n"
                '  "entities": [\n'
                '    {"name": "short entity name (2-4 words max)", "type": "TECHNOLOGY|PROJECT|TOOL|FILE|API|PERSON|PREFERENCE|ERROR|CONCEPT|ORGANIZATION|RULE|DECISION", "description": "brief description"}\n'
                "  ],\n"
                '  "relationships": [\n'
                '    {"subject": "source entity name", "predicate": "USES|DEPENDS_ON|CONFLICTS_WITH|HAS_CONSTRAINT|HAS_VERSION|RUNS_ON|APPLIES_TO|PREVENTS|SUPERSEDES|CAUSED_BY|IMPLEMENTED_IN|STORED_IN|PART_OF|AFFECTS|RELATED_TO|WORKS_ON", "object": "target entity name", "evidence": "brief reason", "confidence": 0.0-1.0}\n'
                "  ]\n"
                "}\n\n"
                "RULES:\n"
                "- Entity names MUST be short (2-4 words). Do NOT use full sentences as names.\n"
                "- If nothing relevant, return {\"entities\": [], \"relationships\": []}."
            )

            try:
                payload = json.dumps({
                    "model": MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "options": {"temperature": 0.1, "num_predict": 1024},
                    "stream": False,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{OLLAMA_HOST}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read().decode("utf-8")
                response = json.loads(body)
                extracted = json.loads(response.get("response", "{}"))
            except Exception as exc:
                print(
                    f"[kg] bootstrap: Ollama error for {file_path} chunk {chunk_idx + 1}: {exc}",
                    file=sys.stderr,
                )
                continue

            for ent in extracted.get("entities", []):
                name = (ent.get("name") or "").strip()
                if not name:
                    continue
                try:
                    upsert_entity(
                        name=name,
                        entity_type=ent.get("type", "CONCEPT"),
                        description=ent.get("description"),
                    )
                    entities_created += 1
                except Exception as exc:
                    print(f"[kg] bootstrap entity error ({name}): {exc}", file=sys.stderr)

            for rel in extracted.get("relationships", []):
                subject = (rel.get("subject") or "").strip()
                obj = (rel.get("object") or "").strip()
                predicate = (rel.get("predicate") or "RELATED_TO").strip()
                if not subject or not obj:
                    continue
                try:
                    result = insert_relationship(
                        subject_name=subject,
                        predicate=predicate,
                        object_name=obj,
                        evidence=rel.get("evidence"),
                        confidence=float(rel.get("confidence", 0.5)),
                    )
                    if result:
                        relationships_created += 1
                except Exception as exc:
                    print(f"[kg] bootstrap rel error ({subject}-{predicate}-{obj}): {exc}", file=sys.stderr)

    # Update kg_sync_state timestamps
    conn = db.get_db()
    now = time.time()
    try:
        for f in file_paths:
            fname = Path(f).name.lower()
            if "claude" in fname:
                conn.execute(
                    "UPDATE kg_sync_state SET value = ?, updated_at = ? WHERE key = 'last_claude_md_sync'",
                    (str(now), now),
                )
            elif "memory" in fname:
                conn.execute(
                    "UPDATE kg_sync_state SET value = ?, updated_at = ? WHERE key = 'last_memory_md_sync'",
                    (str(now), now),
                )
        conn.commit()
    except Exception:
        pass
    conn.close()

    return {
        "entities_created": entities_created,
        "relationships_created": relationships_created,
    }


def detect_communities(max_entities: int = 5000) -> int:
    """Run community detection on the entity-relationship graph.

    Uses NetworkX Louvain if available, falls back to connected-components
    via SQLite recursive CTE. Updates kg_entities.community_id.

    Skips if entity count exceeds max_entities (performance guard).

    Returns the number of communities found.
    """
    conn = db.get_db()

    entity_count = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
    if entity_count == 0:
        conn.close()
        return 0
    if entity_count > max_entities:
        logger.warning(
            "Entity count %d exceeds max_entities %d, skipping community detection",
            entity_count, max_entities,
        )
        conn.close()
        return 0

    now = time.time()

    # Load entities and non-expired relationships
    entities = conn.execute("SELECT id FROM kg_entities").fetchall()
    relationships = conn.execute(
        "SELECT subject_id, object_id, confidence FROM kg_relationships "
        "WHERE valid_until IS NULL OR valid_until > ?",
        (now,),
    ).fetchall()

    entity_ids = [row["id"] for row in entities]

    try:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(entity_ids)
        for rel in relationships:
            G.add_edge(rel["subject_id"], rel["object_id"],
                       weight=rel["confidence"] or 0.5)

        communities = nx.community.louvain_communities(G, weight='weight')

        for community_id, members in enumerate(communities):
            placeholders = ",".join("?" * len(members))
            member_list = list(members)
            conn.execute(
                f"UPDATE kg_entities SET community_id = ? WHERE id IN ({placeholders})",
                [community_id] + member_list,
            )
        num_communities = len(communities)

    except ImportError:
        # Fallback: connected components via recursive CTE
        logger.info("NetworkX unavailable, using CTE fallback for community detection")
        # Build adjacency in Python from the loaded relationships
        adjacency: dict[str, set[str]] = {eid: set() for eid in entity_ids}
        for rel in relationships:
            s, o = rel["subject_id"], rel["object_id"]
            if s in adjacency and o in adjacency:
                adjacency[s].add(o)
                adjacency[o].add(s)

        visited: set[str] = set()
        community_id = 0
        for eid in entity_ids:
            if eid in visited:
                continue
            # BFS to find connected component
            component = []
            queue = [eid]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            placeholders = ",".join("?" * len(component))
            conn.execute(
                f"UPDATE kg_entities SET community_id = ? WHERE id IN ({placeholders})",
                [community_id] + component,
            )
            community_id += 1

        num_communities = community_id

    conn.commit()
    conn.close()
    return num_communities


def apply_relationship_decay() -> dict:
    """Apply time-based decay to KG relationships using kg_decay_config.

    For non-permanent predicates, if age exceeds decay_window_days:
    - Reduce confidence by 50%
    - If confidence drops below 0.1, mark as expired (set valid_until = now)

    Idempotent: runs at most once per 24 hours via kg_sync_state.

    Returns {'decayed': N, 'expired': N}
    """
    conn = db.get_db()
    now = time.time()

    # Check idempotency via kg_sync_state
    row = conn.execute(
        "SELECT value FROM kg_sync_state WHERE key = 'last_relationship_decay'",
    ).fetchone()
    if row:
        try:
            last_run = float(row["value"])
            if (now - last_run) < 86400:
                conn.close()
                return {'decayed': 0, 'expired': 0}
        except (ValueError, TypeError):
            pass

    # Load non-permanent decay config
    configs = conn.execute(
        "SELECT predicate, decay_window_days FROM kg_decay_config "
        "WHERE decay_window_days IS NOT NULL",
    ).fetchall()

    decayed = 0
    expired = 0

    for cfg in configs:
        predicate = cfg["predicate"]
        window_seconds = cfg["decay_window_days"] * 86400
        cutoff = now - window_seconds

        # Find relationships past their decay window that are still active
        rels = conn.execute(
            "SELECT id, confidence FROM kg_relationships "
            "WHERE predicate = ? AND valid_until IS NULL "
            "AND created_at IS NOT NULL AND created_at < ?",
            (predicate, cutoff),
        ).fetchall()

        for rel in rels:
            new_confidence = rel["confidence"] * 0.5
            decayed += 1
            if new_confidence < 0.1:
                conn.execute(
                    "UPDATE kg_relationships SET confidence = ?, valid_until = ? WHERE id = ?",
                    (new_confidence, now, rel["id"]),
                )
                expired += 1
            else:
                conn.execute(
                    "UPDATE kg_relationships SET confidence = ? WHERE id = ?",
                    (new_confidence, rel["id"]),
                )

    # Update sync state
    conn.execute(
        "INSERT OR REPLACE INTO kg_sync_state (key, value, updated_at) "
        "VALUES ('last_relationship_decay', ?, ?)",
        (str(now), now),
    )
    conn.commit()
    conn.close()
    return {'decayed': decayed, 'expired': expired}
