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
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

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

_memories_cache: list[dict] = []
_cache_loaded_at: float = 0.0


def _load_model() -> None:
    global _model, _has_embeddings
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _has_embeddings = True
        print(f"[daemon] Embedding model loaded", flush=True)
    except Exception as exc:
        _model = None
        _has_embeddings = False
        print(f"[daemon] sentence-transformers unavailable ({exc}), using keyword fallback", flush=True)


def _get_embedding(text: str) -> list[float] | None:
    if _model is None:
        return None
    try:
        vec = _model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    now = time.time()
    last = row.get("last_accessed_at") or row.get("created_at", now)
    t_days = max((now - last) / 86400.0, 1e-6)

    stability  = float(row.get("stability", 0.0))
    decay_rate = float(row.get("decay_rate", 0.16))
    lambda_eff = decay_rate * (1.0 - stability * 0.8)
    strength   = math.exp(-lambda_eff * t_days)

    access_count = int(row.get("access_count", 0))
    if access_count == 0:
        return strength * 0.5

    d    = 0.5
    actr = math.log(access_count / (1.0 - d)) - d * math.log(t_days)
    actr_norm = max(0.0, min(1.0, (actr + 5.0) / 10.0))
    return actr_norm * 0.5 + strength * 0.5


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


def _search(query: str, project: str) -> dict:
    """Core search logic. Returns {"hits": [...], "context": "..."}."""
    memories = _load_memories(project)
    if not memories:
        return {"hits": [], "context": ""}

    query_emb = _get_embedding(query) if _has_embeddings else None

    scored = []
    for mem in memories:
        if mem.get("memory_type") not in RETRIEVABLE_TYPES:
            continue
        t_score = _temporal_score(mem)
        if t_score < MIN_TEMPORAL_SCORE:
            continue

        if query_emb is not None and mem.get("embedding") is not None:
            sim = _cosine_similarity(query_emb, mem["embedding"])
            threshold = SIMILARITY_THRESHOLD
        else:
            sim = _keyword_similarity(query, mem.get("content", ""))
            threshold = KEYWORD_THRESHOLD

        if sim < threshold:
            continue

        scored.append({
            **mem,
            "similarity": sim,
            "temporal_score": t_score,
            "final_score": sim * (0.5 + t_score * 0.5),
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    hits = scored[:TOP_K]

    if not hits:
        return {"hits": [], "context": ""}

    _record_access([h["id"] for h in hits])
    # Invalidate cache so next load picks up updated access counts
    global _memories_cache
    _memories_cache = []

    context = _format_context(hits)
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
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/search":
            body    = self._read_body()
            query   = body.get("query", "")
            project = body.get("project", "")
            if not query:
                self._send_json(400, {"error": "query required"})
                return
            result = _search(query, project)
            self._send_json(200, result)

        elif self.path == "/invalidate_cache":
            global _memories_cache, _cache_loaded_at
            _memories_cache = []
            _cache_loaded_at = 0.0
            self._send_json(200, {"invalidated": True})

        else:
            self._send_json(404, {"error": "not found"})


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[daemon] Starting on port {PORT}", flush=True)
    _load_model()

    server = HTTPServer(("127.0.0.1", PORT), _Handler)
    print(f"[daemon] Ready", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"[daemon] Shutting down", flush=True)


if __name__ == "__main__":
    main()
