#!/usr/bin/env python3
"""Phase 4 autonomous integration tests — runs against REAL daemon + REAL DB.

This script validates all Phase 4 features end-to-end:
  1. Importance decay
  2. Near-dedup (Jaccard >= 0.85)
  3. Decisions table + FTS5 BM25 search
  4. RRF fusion in daemon /search
  5. Retrieval feedback logging
  6. FTS5 query sanitization (special chars)
  7. Decision type extraction prompt

Usage: python3 tests/test_phase4_integration.py
Requires: daemon running on port 9876, Ollama not required (tests use DB directly)
"""

import json
import hashlib
import os
import sys
import time
import urllib.request
import urllib.error

# Add hooks dir to path
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TESTS_DIR)
_HOOKS_DIR = os.path.join(_PROJECT_DIR, "hooks")
sys.path.insert(0, _HOOKS_DIR)

import db

DAEMON_URL = "http://127.0.0.1:9876"
TEST_PROJECT = "/test/phase4/integration"
TEST_SESSION = "phase4-integration-test"

# ── Helpers ──────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0
_errors = []


def _result(name, ok, detail=""):
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        _errors.append((name, detail))
        print(f"  [FAIL] {name}: {detail}")


def _daemon_post(path, body):
    """POST to daemon, return parsed JSON."""
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{DAEMON_URL}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _daemon_get(path):
    """GET from daemon, return parsed JSON."""
    req = urllib.request.Request(f"{DAEMON_URL}{path}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _insert_test_memory(content, memory_type="semantic", importance=6,
                        decision_type=None, subject=None, predicate=None):
    """Insert a memory directly into DB and return its id."""
    mem_dict = {
        "content": content,
        "type": memory_type,
        "importance": importance,
        "subject": subject,
        "predicate": predicate,
    }
    mem_id = db.insert_memory(mem_dict, TEST_SESSION, TEST_PROJECT)

    # If decision_type, also insert into decisions table
    if decision_type:
        db.insert_decision(
            memory_id=mem_id,
            decision_type=decision_type,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            keywords=content.lower().split()[:5],
            project=TEST_PROJECT,
            session_id=TEST_SESSION,
        )

    return mem_id


def _get_memory(mem_id):
    """Fetch a single memory row by id."""
    conn = db.get_db()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _count_decisions():
    """Count decisions for test project."""
    conn = db.get_db()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM decisions WHERE project = ?",
        (TEST_PROJECT,),
    ).fetchone()
    conn.close()
    return row["cnt"]


def _cleanup():
    """Remove test data."""
    conn = db.get_db()
    # Get test memory ids
    rows = conn.execute(
        "SELECT id FROM memories WHERE project = ?", (TEST_PROJECT,)
    ).fetchall()
    test_ids = [r["id"] for r in rows]

    if test_ids:
        placeholders = ",".join("?" * len(test_ids))
        conn.execute(f"DELETE FROM decisions WHERE memory_id IN ({placeholders})", test_ids)
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", test_ids)
        conn.commit()

    conn.close()

    # Invalidate daemon cache
    try:
        _daemon_post("/invalidate_cache", {})
    except Exception:
        pass


# ── Test 1: Daemon Health ────────────────────────────────────────────────────

def test_daemon_health():
    print("\n── Test 1: Daemon Health ──")
    try:
        result = _daemon_get("/health")
        _result("daemon is healthy", result.get("status") == "ok",
                f"got: {result}")
    except Exception as e:
        _result("daemon is healthy", False, str(e))


# ── Test 2: Importance Decay ─────────────────────────────────────────────────

def test_importance_decay():
    print("\n── Test 2: Importance Decay ──")

    # Insert memory with high importance, backdate it
    mem_id = _insert_test_memory(
        "Phase4 test: always use pytest for Python testing",
        memory_type="procedural", importance=8
    )

    # Backdate to 40 days ago
    conn = db.get_db()
    past = time.time() - (40 * 86400)
    conn.execute(
        "UPDATE memories SET created_at = ?, last_accessed_at = NULL WHERE id = ?",
        (past, mem_id),
    )
    # Reset the decay timestamp so decay will run
    conn.execute(
        "DELETE FROM kg_sync_state WHERE key = 'last_decay_run'"
    )
    conn.commit()
    conn.close()

    # Run decay
    decayed = db.decay_stale_importance(days_threshold=30, floor=3)
    _result("decay ran successfully", decayed >= 1,
            f"decayed {decayed} memories")

    # Verify importance decreased
    mem = _get_memory(mem_id)
    _result("importance decreased by 1", mem["importance"] == 7,
            f"expected 7, got {mem['importance']}")

    # Run decay again immediately — idempotency guard should block
    decayed2 = db.decay_stale_importance(days_threshold=30, floor=3)
    _result("idempotency guard blocks second run", decayed2 == 0,
            f"expected 0, got {decayed2}")

    mem2 = _get_memory(mem_id)
    _result("importance unchanged after second run", mem2["importance"] == 7,
            f"expected 7, got {mem2['importance']}")


# ── Test 3: Near-Dedup (Jaccard) ─────────────────────────────────────────────

def test_near_dedup():
    print("\n── Test 3: Near-Dedup (Jaccard >= 0.85) ──")

    # Insert original
    orig_id = _insert_test_memory(
        "Use Redis Stack for cache layer not plain Redis or MySQL",
        memory_type="correction", importance=7
    )

    # Insert near-duplicate (same words, slightly rephrased)
    dup_id = _insert_test_memory(
        "Use Redis Stack for cache layer not plain Redis or MySQL database",
        memory_type="correction", importance=7
    )

    _result("near-dup returns same id", orig_id == dup_id,
            f"orig={orig_id[:8]}, dup={dup_id[:8]}")

    # Verify access count bumped
    mem = _get_memory(orig_id)
    _result("access_count incremented", mem["access_count"] >= 1,
            f"access_count={mem['access_count']}")

    # Insert genuinely different content — should get new id
    diff_id = _insert_test_memory(
        "Use PostgreSQL for production relational database",
        memory_type="correction", importance=7
    )
    _result("different content gets new id", diff_id != orig_id,
            f"diff={diff_id[:8]}, orig={orig_id[:8]}")


# ── Test 4: Decisions Table + FTS5 BM25 ──────────────────────────────────────

def test_decisions():
    print("\n── Test 4: Decisions Table + FTS5 BM25 ──")

    before_count = _count_decisions()

    # Insert memories with decision types
    _insert_test_memory(
        "Use SQLite for all structured storage in ensemble memory system",
        decision_type="ARCHITECTURAL", importance=8
    )
    _insert_test_memory(
        "Prefer 4 spaces indentation for all Python code following PEP 8",
        decision_type="PREFERENCE", importance=6
    )
    _insert_test_memory(
        "Redis connection timeout should be set to 5 seconds maximum",
        decision_type="CONSTRAINT", importance=7
    )

    after_count = _count_decisions()
    _result("3 decisions inserted", after_count - before_count == 3,
            f"before={before_count}, after={after_count}")

    # BM25 search
    results = db.search_decisions_bm25("SQLite storage", project=TEST_PROJECT)
    _result("BM25 finds SQLite decision", len(results) >= 1,
            f"got {len(results)} results")

    if results:
        _result("correct decision type returned",
                results[0]["decision_type"] == "ARCHITECTURAL",
                f"got {results[0].get('decision_type')}")

    # BM25 search for Python
    results2 = db.search_decisions_bm25("Python indentation PEP",
                                         project=TEST_PROJECT)
    _result("BM25 finds Python preference", len(results2) >= 1,
            f"got {len(results2)} results")

    # BM25 with no matches
    results3 = db.search_decisions_bm25("quantum_computing_xyz",
                                         project=TEST_PROJECT)
    _result("no false positives", len(results3) == 0,
            f"got {len(results3)} results")


# ── Test 5: RRF Fusion in Daemon /search ─────────────────────────────────────

def test_rrf_search():
    print("\n── Test 5: RRF Fusion in Daemon /search ──")

    # Insert a distinctive memory with embedding
    mem_id = _insert_test_memory(
        "Zephyr quantum processor requires helium cooling at minus 273 celsius",
        memory_type="semantic", importance=7
    )

    # Generate embedding via daemon
    try:
        embed_result = _daemon_post("/embed", {
            "text": "Zephyr quantum processor requires helium cooling at minus 273 celsius"
        })
        if embed_result.get("embedding"):
            db.store_embedding(mem_id, embed_result["embedding"])
    except Exception:
        pass  # Embedding is best-effort

    # Invalidate cache so daemon picks up new memory
    _daemon_post("/invalidate_cache", {})

    # Search — should find via cosine similarity and/or BM25
    result = _daemon_post("/search", {
        "query": "Zephyr quantum cooling helium",
        "project": TEST_PROJECT,
    })

    hits = result.get("hits", [])
    _result("RRF search returns results", len(hits) >= 1,
            f"got {len(hits)} hits")

    if hits:
        # Check that our test memory is in results
        found = any("Zephyr" in h.get("content", "") for h in hits)
        _result("correct memory found", found,
                f"contents: {[h.get('content', '')[:40] for h in hits]}")

        # Check RRF score is present
        has_rrf = "rrf_score" in hits[0]
        _result("rrf_score field present", has_rrf,
                f"keys: {list(hits[0].keys())}")

        # Check final_score is present
        has_final = "final_score" in hits[0]
        _result("final_score field present", has_final,
                f"keys: {list(hits[0].keys())}")

    context = result.get("context", "")
    _result("context string generated", len(context) > 0,
            f"context length: {len(context)}")


# ── Test 6: Retrieval Feedback Logging ───────────────────────────────────────

def test_feedback_logging():
    print("\n── Test 6: Retrieval Feedback Logging ──")

    result = _daemon_post("/log_feedback", {
        "query": "Phase4 integration test query",
        "type": "miss",
        "session_id": TEST_SESSION,
    })

    _result("/log_feedback returns success", result.get("logged") is True,
            f"got: {result}")

    # Check the JSONL file was written
    log_file = os.path.expanduser("~/.ensemble_memory/logs/retrieval_feedback.jsonl")
    _result("feedback log file exists", os.path.exists(log_file),
            f"path: {log_file}")

    if os.path.exists(log_file):
        with open(log_file) as f:
            lines = f.readlines()
        last_line = json.loads(lines[-1])
        _result("log entry has correct query",
                "Phase4 integration" in last_line.get("query", ""),
                f"got: {last_line.get('query', '')[:50]}")


# ── Test 7: FTS5 Special Character Handling ──────────────────────────────────

def test_fts5_special_chars():
    print("\n── Test 7: FTS5 Special Character Handling ──")

    # These queries contain FTS5 special chars that would crash without sanitization
    dangerous_queries = [
        'what\'s the "config" setting?',
        "use NOT this OR that",
        "NEAR(test, 5)",
        "column:value",
        "*wildcard*",
        "",  # empty query
    ]

    for q in dangerous_queries:
        try:
            results = db.search_decisions_bm25(q, project=TEST_PROJECT)
            _result(f"safe query: {q[:30]!r}", True)
        except Exception as e:
            _result(f"safe query: {q[:30]!r}", False, str(e))

    # Also test through daemon /search (hits the daemon's _bm25_search)
    try:
        result = _daemon_post("/search", {
            "query": 'what\'s the "config"?',
            "project": TEST_PROJECT,
        })
        _result("daemon handles special chars",
                isinstance(result.get("hits"), list),
                f"got: {type(result.get('hits'))}")
    except Exception as e:
        _result("daemon handles special chars", False, str(e))


# ── Test 8: Decision Type Normalization ──────────────────────────────────────

def test_decision_type_normalization():
    print("\n── Test 8: Decision Type Normalization ──")

    mem_id = _insert_test_memory(
        "Phase4 normalization test memory for type validation",
        importance=6
    )

    # Valid type
    d1 = db.insert_decision(
        memory_id=mem_id, decision_type="ARCHITECTURAL",
        content_hash="norm_test_1", project=TEST_PROJECT, session_id=TEST_SESSION
    )
    _result("valid type accepted", d1 is not None)

    # Pipe-separated (LLM quirk)
    d2 = db.insert_decision(
        memory_id=mem_id, decision_type="PREFERENCE | CONSTRAINT",
        content_hash="norm_test_2", project=TEST_PROJECT, session_id=TEST_SESSION
    )
    _result("pipe-separated normalized", d2 is not None)

    # Verify it stored as PREFERENCE (first valid)
    if d2:
        conn = db.get_db()
        row = conn.execute("SELECT decision_type FROM decisions WHERE id = ?", (d2,)).fetchone()
        conn.close()
        _result("normalized to first valid type",
                row["decision_type"] == "PREFERENCE",
                f"got: {row['decision_type']}")

    # Invalid type
    d3 = db.insert_decision(
        memory_id=mem_id, decision_type="INVALID_GARBAGE",
        content_hash="norm_test_3", project=TEST_PROJECT, session_id=TEST_SESSION
    )
    _result("invalid type returns None", d3 is None)


# ── Test 9: End-to-End Write → Search Roundtrip ─────────────────────────────

def test_e2e_roundtrip():
    print("\n── Test 9: End-to-End Write → Search Roundtrip ──")

    # Insert a unique decision
    unique_content = f"Moonbeam protocol v{int(time.time())} requires titanium shielding"
    mem_id = _insert_test_memory(
        unique_content,
        memory_type="semantic",
        importance=8,
        decision_type="CONSTRAINT",
    )

    # Embed it
    try:
        embed_result = _daemon_post("/embed", {"text": unique_content})
        if embed_result.get("embedding"):
            db.store_embedding(mem_id, embed_result["embedding"])
    except Exception:
        pass

    # Invalidate cache
    _daemon_post("/invalidate_cache", {})

    # Search via daemon — should find via both cosine AND BM25
    result = _daemon_post("/search", {
        "query": "Moonbeam titanium shielding",
        "project": TEST_PROJECT,
    })

    hits = result.get("hits", [])
    found = any("Moonbeam" in h.get("content", "") for h in hits)
    _result("e2e: inserted memory found via search", found,
            f"hits={len(hits)}, contents={[h.get('content','')[:30] for h in hits]}")

    # Also search via BM25 in decisions
    bm25_results = db.search_decisions_bm25("Moonbeam titanium", project=TEST_PROJECT)
    _result("e2e: found via BM25 decisions search", len(bm25_results) >= 1,
            f"bm25 results: {len(bm25_results)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 4 Integration Tests — Real Daemon + Real DB")
    print("=" * 60)

    # Check daemon is up
    try:
        _daemon_get("/health")
    except Exception:
        print("\nERROR: Daemon not running on port 9876")
        print("Start it with: bash daemon/daemon_ctl.sh start")
        sys.exit(1)

    # Clean up any leftover test data
    _cleanup()

    try:
        test_daemon_health()
        test_importance_decay()
        test_near_dedup()
        test_decisions()
        test_rrf_search()
        test_feedback_logging()
        test_fts5_special_chars()
        test_decision_type_normalization()
        test_e2e_roundtrip()
    finally:
        # Clean up test data
        _cleanup()

    print("\n" + "=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 60)

    if _errors:
        print("\nFailed tests:")
        for name, detail in _errors:
            print(f"  - {name}: {detail}")

    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
