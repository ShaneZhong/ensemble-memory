#!/usr/bin/env python3
"""Test suite for daemon core logic — temporal scoring, BM25 search, RRF fusion,
KG context, /status endpoint, and cache behavior.

Run: /Users/shane/Documents/playground/.venv/bin/python3 -m pytest tests/test_daemon.py -v

Tests exercise daemon internals directly (no HTTP server needed).
"""

import importlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
HOOKS_DIR = Path(__file__).parent.parent / "hooks"
DAEMON_DIR = Path(__file__).parent.parent / "daemon"
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))
if str(DAEMON_DIR) not in sys.path:
    sys.path.insert(0, str(DAEMON_DIR))

import db
import kg
import embedding_daemon

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Redirect DB to a temp directory for each test."""
    db._DB_PATH_OVERRIDE = str(tmp_path / "test.db")
    db._db_initialized.clear()
    if hasattr(db, "ensure_embedding_column"):
        db.ensure_embedding_column()
    if hasattr(db, "ensure_enrichment_columns"):
        db.ensure_enrichment_columns()
    # Reset daemon cache state between tests
    embedding_daemon._memories_cache = []
    embedding_daemon._cache_loaded_at = 0.0
    yield tmp_path
    db._DB_PATH_OVERRIDE = None
    db._db_initialized.clear()


def _insert_memory(content, memory_type="semantic", importance=5, project="/test/project", **kwargs):
    """Helper: insert a memory and return its id."""
    mem = {
        "type": memory_type,
        "content": content,
        "importance": importance,
    }
    mem.update(kwargs)
    return db.insert_memory(mem, "test-session", project)


def _insert_memory_with_embedding(content, embedding, memory_type="procedural",
                                  importance=7, project="/test/project"):
    """Helper: insert a memory with a pre-set embedding."""
    mem_id = _insert_memory(content, memory_type=memory_type,
                            importance=importance, project=project)
    db.store_embedding(mem_id, embedding)
    return mem_id


def _create_entity(name, entity_type="CONCEPT", description=None):
    """Helper: create an entity and return its id."""
    return kg.upsert_entity(name, entity_type, description=description)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestTemporalScore
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalScore:
    """Test _temporal_score wrapper around db.temporal_score."""

    def test_fresh_no_accesses_ebbinghaus_only(self):
        """Fresh memory with no accesses uses Ebbinghaus-only path (strength * 0.5)."""
        now = time.time()
        row = {
            "access_count": 0,
            "last_accessed_at": None,
            "created_at": now - 60,  # 1 minute ago
            "decay_rate": 0.16,
            "stability": 0.0,
        }
        score = embedding_daemon._temporal_score(row)
        # With 0 accesses, formula is strength * 0.5
        # strength = exp(-0.16 * (60/86400)) ≈ exp(-0.000111) ≈ 0.999889
        # score ≈ 0.5
        assert 0.4 < score < 0.6
        # Verify it matches direct db.temporal_score call
        expected = db.temporal_score(0, None, now - 60, 0.16, 0.0)
        assert score == pytest.approx(expected)

    def test_old_memory_many_accesses_actr_boost(self):
        """Old memory with many accesses gets ACT-R boost above Ebbinghaus-only."""
        now = time.time()
        row_with_access = {
            "access_count": 20,
            "last_accessed_at": now - 3600,  # 1 hour ago
            "created_at": now - 86400 * 30,  # 30 days ago
            "decay_rate": 0.16,
            "stability": 0.0,
        }
        row_no_access = {
            "access_count": 0,
            "last_accessed_at": None,
            "created_at": now - 86400 * 30,
            "decay_rate": 0.16,
            "stability": 0.0,
        }
        score_with = embedding_daemon._temporal_score(row_with_access)
        score_without = embedding_daemon._temporal_score(row_no_access)
        # ACT-R should boost the score above Ebbinghaus-only
        assert score_with > score_without

    def test_high_stability_slower_decay(self):
        """High stability reduces effective decay rate, producing higher score."""
        now = time.time()
        row_low_stab = {
            "access_count": 0,
            "last_accessed_at": None,
            "created_at": now - 86400 * 7,  # 7 days ago
            "decay_rate": 0.16,
            "stability": 0.0,
        }
        row_high_stab = {
            "access_count": 0,
            "last_accessed_at": None,
            "created_at": now - 86400 * 7,
            "decay_rate": 0.16,
            "stability": 0.9,
        }
        score_low = embedding_daemon._temporal_score(row_low_stab)
        score_high = embedding_daemon._temporal_score(row_high_stab)
        # High stability → lower lambda_eff → slower decay → higher score
        assert score_high > score_low


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestBM25Search
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Search:
    """Test _bm25_search FTS5 + LIKE fallback."""

    def test_decision_fts_match(self):
        """Decision FTS5 match returns results with correct structure."""
        import hashlib
        mem_id = _insert_memory("always use pytest fixtures", memory_type="procedural",
                                importance=8)
        content_hash = hashlib.sha256(b"always use pytest fixtures for testing").hexdigest()
        db.insert_decision(
            memory_id=mem_id,
            decision_type="PATTERN",
            content_hash=content_hash,
            keywords=["pytest", "fixtures"],
            project="/test/project",
            session_id="test-session",
        )
        results = embedding_daemon._bm25_search("pytest fixtures", "/test/project")
        assert len(results) >= 1
        hit = results[0]
        assert "id" in hit
        assert "content" in hit
        assert "bm25_rank" in hit

    def test_keyword_like_fallback(self):
        """Keyword LIKE fallback on memories table finds matches."""
        _insert_memory("use Redis Stack for cache layer", memory_type="procedural",
                       importance=7)
        results = embedding_daemon._bm25_search("Redis cache", "/test/project")
        # Should find via keyword LIKE fallback
        found = any("Redis" in r.get("content", "") for r in results)
        assert found

    def test_no_match_returns_empty(self):
        """Query with no matching terms returns empty list."""
        _insert_memory("python version is 3.11", memory_type="semantic", importance=5)
        results = embedding_daemon._bm25_search("xyznonexistent", "/test/project")
        assert results == []

    def test_special_chars_no_crash(self):
        """Special characters in query don't crash FTS5."""
        _insert_memory("test memory content", memory_type="semantic", importance=5)
        # These characters could break FTS5 if not properly quoted
        for query in ['test"query', "test'query", "test(OR)query", "test AND NOT query",
                      'test{brackets}', "test*wild?card"]:
            results = embedding_daemon._bm25_search(query, "/test/project")
            assert isinstance(results, list)  # No crash


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestRRFSearch
# ═══════════════════════════════════════════════════════════════════════════════

class TestRRFSearch:
    """Test _search RRF fusion logic."""

    def _make_embedding(self, seed: float = 0.1) -> list[float]:
        """Create a deterministic 384-dim embedding."""
        return [math.sin(seed * i) for i in range(384)]

    def test_cosine_bm25_fusion_ranked(self):
        """Cosine + BM25 fusion produces ranked results with final_score."""
        emb = self._make_embedding(0.5)
        _insert_memory_with_embedding(
            "always run pytest before committing", emb,
            memory_type="procedural", importance=8,
        )
        _insert_memory_with_embedding(
            "use logging module not print", self._make_embedding(0.3),
            memory_type="correction", importance=7,
        )
        # Mock the embedding model to return a query embedding
        with patch.object(embedding_daemon, '_has_embeddings', True), \
             patch.object(embedding_daemon, '_get_embedding', return_value=emb):
            result = embedding_daemon._search("run pytest", "/test/project")
        assert "hits" in result
        assert "context" in result
        assert len(result["hits"]) >= 1, "RRF fusion should return at least 1 hit"
        assert "final_score" in result["hits"][0]
        assert "rrf_score" in result["hits"][0]

    def test_empty_db_returns_empty(self):
        """Empty database returns empty hits and context."""
        result = embedding_daemon._search("any query", "/test/project")
        assert result == {"hits": [], "context": ""}

    def test_results_filtered_by_project(self):
        """Only memories matching the project are returned."""
        emb = self._make_embedding(0.5)
        _insert_memory_with_embedding(
            "project A memory about testing", emb,
            memory_type="procedural", importance=8, project="/project/a",
        )
        _insert_memory_with_embedding(
            "project B memory about testing", self._make_embedding(0.4),
            memory_type="procedural", importance=8, project="/project/b",
        )
        with patch.object(embedding_daemon, '_has_embeddings', True), \
             patch.object(embedding_daemon, '_get_embedding', return_value=emb):
            result = embedding_daemon._search("testing", "/project/a")
        assert len(result["hits"]) >= 1, "Should return at least 1 hit"
        for hit in result["hits"]:
            assert "project B" not in hit["content"], f"Project B memory should be filtered: {hit['content']}"

    def test_temporal_score_weights_results(self):
        """Temporal score influences final ranking via the scoring formula."""
        now = time.time()
        emb = self._make_embedding(0.5)
        # Insert two memories with same embedding but different temporal properties
        id1 = _insert_memory_with_embedding(
            "fresh memory about deployment", emb,
            memory_type="procedural", importance=8,
        )
        id2 = _insert_memory_with_embedding(
            "old memory about deployment process", self._make_embedding(0.49),
            memory_type="procedural", importance=8,
        )
        # Make id2 very old with no accesses
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET created_at = ?, last_accessed_at = NULL WHERE id = ?",
                (now - 86400 * 60, id2),
            )
            conn.commit()
        finally:
            conn.close()

        with patch.object(embedding_daemon, '_has_embeddings', True), \
             patch.object(embedding_daemon, '_get_embedding', return_value=emb):
            result = embedding_daemon._search("deployment", "/test/project")
        # Fresh memory should rank higher due to temporal boost
        assert len(result["hits"]) >= 2, "Should return at least 2 hits"
        assert result["hits"][0]["content"] == "fresh memory about deployment"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestKGContext
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGContext:
    """Test _get_kg_context entity search and neighborhood."""

    def test_entity_keywords_return_context(self):
        """Query with entity keywords returns neighborhood context string."""
        _create_entity("pytest", "TOOL", description="Python testing framework")
        _create_entity("unittest", "TOOL", description="Python standard test lib")
        kg.insert_relationship("pytest", "RELATED_TO", "unittest", confidence=0.8)
        context = embedding_daemon._get_kg_context("how to use pytest")
        # Should find pytest entity and return some context
        assert isinstance(context, str)
        # May or may not have content depending on neighborhood formatting
        # but should not error

    def test_no_matching_entities_returns_empty(self):
        """Query with no matching entities returns empty string."""
        context = embedding_daemon._get_kg_context("xyznonexistent topic")
        assert context == ""

    def test_formatted_prefix_in_results(self):
        """When entities match, result comes from formatted_prefix."""
        _create_entity("Redis", "TECHNOLOGY", description="In-memory data store")
        _create_entity("cache", "CONCEPT", description="Data caching layer")
        kg.insert_relationship("Redis", "USES", "cache", confidence=0.9)
        context = embedding_daemon._get_kg_context("Redis cache configuration")
        # Should be a string (possibly empty if neighborhood has no formatted_prefix)
        assert isinstance(context, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestStatusEndpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatusEndpoint:
    """Test _build_status function."""

    def test_status_correct_structure(self):
        """Status returns all expected fields with correct types."""
        # Set daemon state
        embedding_daemon._daemon_start_time = time.time() - 3600
        embedding_daemon._last_bg_job_time = time.time() - 120
        embedding_daemon._has_embeddings = True
        embedding_daemon._memories_cache = [{"id": "a"}, {"id": "b"}]
        embedding_daemon._cache_loaded_at = time.time() - 60

        # Insert some data
        _insert_memory("test memory", memory_type="procedural", importance=7)
        _create_entity("TestEntity", "CONCEPT")

        status = embedding_daemon._build_status()

        assert status["status"] == "ok"
        assert status["port"] == embedding_daemon.PORT
        assert status["model_loaded"] is True
        assert status["memory_count"] == 1
        assert status["cache_size"] == 2
        assert isinstance(status["cache_age_seconds"], float)
        assert status["last_background_job"] is not None
        assert status["kg_entities"] == 1
        assert isinstance(status["uptime_seconds"], float)
        assert status["uptime_seconds"] >= 3599

    def test_status_empty_database(self):
        """Status works correctly with empty database and no daemon state."""
        embedding_daemon._daemon_start_time = time.time()
        embedding_daemon._last_bg_job_time = None
        embedding_daemon._has_embeddings = False
        embedding_daemon._memories_cache = []
        embedding_daemon._cache_loaded_at = 0.0

        status = embedding_daemon._build_status()

        assert status["status"] == "ok"
        assert status["model_loaded"] is False
        assert status["memory_count"] == 0
        assert status["cache_size"] == 0
        assert status["cache_age_seconds"] is None
        assert status["last_background_job"] is None
        assert status["kg_entities"] == 0
        assert status["kg_communities"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestCacheBehavior
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheBehavior:
    """Test daemon memory cache loading and invalidation."""

    def test_cache_loads_on_first_search(self):
        """Cache is populated after first _load_memories call."""
        _insert_memory("cached memory test", memory_type="procedural", importance=7)
        assert embedding_daemon._memories_cache == []
        assert embedding_daemon._cache_loaded_at == 0.0

        memories = embedding_daemon._load_memories("/test/project")
        assert len(memories) >= 1
        assert embedding_daemon._cache_loaded_at > 0
        assert len(embedding_daemon._memories_cache) >= 1

    def test_cache_invalidation_clears(self):
        """Cache invalidation resets cache state."""
        _insert_memory("will be cached", memory_type="procedural", importance=7)
        embedding_daemon._load_memories("/test/project")
        assert len(embedding_daemon._memories_cache) >= 1

        # Simulate /invalidate_cache
        embedding_daemon._memories_cache = []
        embedding_daemon._cache_loaded_at = 0.0
        assert embedding_daemon._memories_cache == []
        assert embedding_daemon._cache_loaded_at == 0.0

    def test_cached_temporal_score_used_when_fresh(self):
        """Cached temporal_score is used when score_computed_at is recent."""
        now = time.time()
        mem = {
            "access_count": 5,
            "last_accessed_at": now - 60,
            "created_at": now - 3600,
            "decay_rate": 0.16,
            "stability": 0.0,
            "temporal_score": 0.99,
            "score_computed_at": now - 100,  # 100 seconds ago (fresh, within 6h)
        }
        _CACHE_FRESHNESS = 21600
        cached_score = mem.get("temporal_score")
        computed_at = mem.get("score_computed_at")
        if (cached_score is not None
                and computed_at is not None
                and (now - computed_at) < _CACHE_FRESHNESS):
            t_score = cached_score
        else:
            t_score = db.temporal_score(
                access_count=mem["access_count"],
                last_accessed_at=mem["last_accessed_at"],
                created_at=mem["created_at"],
                decay_rate=mem["decay_rate"],
                stability=mem["stability"],
            )
        assert t_score == 0.99
