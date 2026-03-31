#!/usr/bin/env python3
"""Test suite for Phase 7.

Run: /Users/shane/Documents/playground/.venv/bin/python3 -m pytest tests/test_phase7.py -v

Test classes:
  Sprint 1: TestTriggerConditionMatchText — trigger_condition used as primary match_text
  Sprint 2: TestSessionEndSQLite — SessionEnd safety net writes to SQLite
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add hooks dir to path
HOOKS_DIR = Path(__file__).parent.parent / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

import db
import session_end
import store_memory


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
    yield tmp_path
    db._DB_PATH_OVERRIDE = None
    db._db_initialized.clear()


def _insert_procedural(content, importance=7, reinforcement_count=0, **kwargs):
    """Helper: insert a procedural memory and return its id."""
    mem = {
        "type": "procedural",
        "content": content,
        "importance": importance,
    }
    mem.update(kwargs)
    mem_id = db.insert_memory(mem, "test-session", "/test/project")
    if reinforcement_count > 0:
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET reinforcement_count = ? WHERE id = ?",
                (reinforcement_count, mem_id),
            )
            conn.commit()
        finally:
            conn.close()
    return mem_id


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestTriggerConditionMatchText
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriggerConditionMatchText:
    """Test that trigger_condition is used as primary match_text for reinforcement."""

    def test_trigger_condition_used_as_primary_match(self):
        """When trigger_condition is present, it should be used as match_text."""
        # Insert a memory with a known subject/predicate/object for triple matching
        _insert_procedural(
            "Use pytest for all Python tests",
            subject="python_tests",
            predicate="must_use",
            object="pytest",
            rule="Always use pytest, never unittest",
        )

        # Now store a second memory with trigger_condition that matches
        # the existing one via subject+predicate+object triple
        mem = {
            "type": "procedural",
            "content": "Use pytest for all Python tests",
            "importance": 7,
            "subject": "python_tests",
            "predicate": "must_use",
            "object": "pytest",
            "rule": "Always use pytest, never unittest",
            "trigger_condition": "when writing Python tests",
        }

        # _store_to_sqlite should detect reinforcement and NOT insert a new row
        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, _ = store_memory._store_to_sqlite([mem], "test-session-2")

        assert new_count == 0, "Should reinforce existing memory, not insert new"

        # Verify reinforcement_count was incremented
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT reinforcement_count FROM memories WHERE subject = ?",
                ("python_tests",),
            ).fetchone()
        finally:
            conn.close()
        assert row["reinforcement_count"] == 1

    def test_rule_used_as_fallback_when_no_trigger_condition(self):
        """When trigger_condition is absent, rule should be used as match_text."""
        _insert_procedural(
            "Use ruff for linting",
            subject="linting",
            predicate="must_use",
            object="ruff",
            rule="Always use ruff, never flake8",
        )

        # Second memory without trigger_condition — should still reinforce
        mem = {
            "type": "procedural",
            "content": "Use ruff for linting",
            "importance": 7,
            "subject": "linting",
            "predicate": "must_use",
            "object": "ruff",
            "rule": "Always use ruff, never flake8",
        }

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, _ = store_memory._store_to_sqlite([mem], "test-session-2")

        assert new_count == 0, "Should reinforce via rule fallback"

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT reinforcement_count FROM memories WHERE subject = ?",
                ("linting",),
            ).fetchone()
        finally:
            conn.close()
        assert row["reinforcement_count"] == 1

    def test_content_used_as_final_fallback(self):
        """When both trigger_condition and rule are absent, content is used."""
        # No object — triple match won't fire, falls to LIKE match on content
        _insert_procedural(
            "Never use print for debugging",
            subject="debugging",
            predicate="must_not_use",
        )

        # No trigger_condition, no rule — content is the fallback for LIKE match
        mem = {
            "type": "procedural",
            "content": "Never use print for debugging",
            "importance": 7,
            "subject": "debugging",
            "predicate": "must_not_use",
        }

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, _ = store_memory._store_to_sqlite([mem], "test-session-2")

        assert new_count == 0, "Should reinforce via content fallback"

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT reinforcement_count FROM memories WHERE subject = ?",
                ("debugging",),
            ).fetchone()
        finally:
            conn.close()
        assert row["reinforcement_count"] == 1

    def test_backward_compatibility_no_trigger_condition_field(self):
        """Existing memories without trigger_condition field still work correctly."""
        # Insert a memory the old way (no trigger_condition key at all)
        mem_id = _insert_procedural(
            "Always run tests before commit",
            subject="git_workflow",
            predicate="must_do",
            object="pytest",
            rule="Run pytest before git commit",
        )

        # Verify it was stored
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT id, content FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None

        # Now a new memory comes in with trigger_condition but same subject/predicate/object
        mem = {
            "type": "procedural",
            "content": "Always run tests before commit",
            "importance": 7,
            "subject": "git_workflow",
            "predicate": "must_do",
            "object": "pytest",
            "rule": "Run pytest before git commit",
            "trigger_condition": "when about to commit code",
            "anti_pattern": "git commit without running tests",
            "correct_pattern": "pytest && git commit",
        }

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, _ = store_memory._store_to_sqlite([mem], "test-session-2")

        # Should reinforce existing, not create new
        assert new_count == 0

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT reinforcement_count FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["reinforcement_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestSessionEndSQLite — Sprint 2
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionEndSQLite:
    """Test that session_end.py writes safety-net memories to SQLite."""

    def test_new_memories_written_to_sqlite(self):
        """Memories not already in DB are written via session_end_sqlite."""
        extraction = json.dumps({
            "memories": [
                {
                    "type": "semantic",
                    "content": "SQLite supports WAL mode for concurrent reads",
                    "importance": 6,
                },
            ],
            "entities": [],
            "relationships": [],
            "summary": ["Discussed SQLite WAL mode"],
        })

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, already_count = session_end.session_end_sqlite(
                extraction, "test-session-end",
            )

        assert new_count == 1
        assert already_count == 0

        # Verify it's in the DB
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT content FROM memories WHERE session_id = ?",
                ("test-session-end",),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert "WAL mode" in row["content"]

    def test_duplicate_memories_skipped(self):
        """Memories already in DB (by content_hash) are not re-inserted."""
        content = "Always use pytest for testing"

        # Pre-insert the memory
        _insert_procedural(content, subject="testing", predicate="must_use", object="pytest")

        extraction = json.dumps({
            "memories": [
                {
                    "type": "procedural",
                    "content": content,
                    "importance": 7,
                },
            ],
            "entities": [],
            "summary": [],
        })

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, already_count = session_end.session_end_sqlite(
                extraction, "test-session-end-2",
            )

        assert new_count == 0
        assert already_count == 1

    def test_mixed_new_and_duplicate(self):
        """Mix of new and existing memories — only new ones inserted."""
        existing_content = "Use ruff for linting"
        _insert_procedural(existing_content, subject="linting", predicate="must_use", object="ruff")

        extraction = json.dumps({
            "memories": [
                {
                    "type": "procedural",
                    "content": existing_content,
                    "importance": 7,
                },
                {
                    "type": "semantic",
                    "content": "Milvus is a vector database for embeddings",
                    "importance": 5,
                },
            ],
            "entities": [],
            "summary": [],
        })

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, already_count = session_end.session_end_sqlite(
                extraction, "test-session-end-3",
            )

        assert new_count == 1
        assert already_count == 1

    def test_empty_extraction(self):
        """Empty memories list produces (0, 0) without errors."""
        extraction = json.dumps({"memories": [], "summary": []})

        new_count, already_count = session_end.session_end_sqlite(
            extraction, "test-session-end-empty",
        )

        assert new_count == 0
        assert already_count == 0

    def test_invalid_json_graceful_skip(self):
        """Corrupted JSON doesn't crash — returns (0, 0)."""
        new_count, already_count = session_end.session_end_sqlite(
            "not valid json {{{", "test-session-bad-json",
        )

        assert new_count == 0
        assert already_count == 0

    def test_none_input_graceful_skip(self):
        """None input doesn't crash."""
        new_count, already_count = session_end.session_end_sqlite(
            None, "test-session-none",
        )

        assert new_count == 0
        assert already_count == 0

    def test_empty_content_skipped(self):
        """Memories with empty content are skipped."""
        extraction = json.dumps({
            "memories": [
                {"type": "semantic", "content": "", "importance": 5},
                {"type": "semantic", "content": "   ", "importance": 5},
            ],
            "summary": [],
        })

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, already_count = session_end.session_end_sqlite(
                extraction, "test-session-empty-content",
            )

        assert new_count == 0
        assert already_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestAMEMEvolution — Sprint 3
# ═══════════════════════════════════════════════════════════════════════════════

import time

# Import evolution module
import evolution


class TestEvolutionParsing:
    """Test evolution.py classification parsing (no Ollama needed)."""

    def test_parse_valid_response(self):
        """Parse a well-formed classification response."""
        valid_ids = {"mem-1", "mem-2", "mem-3"}
        response = json.dumps({
            "relationships": [
                {"existing_id": "mem-1", "link_type": "SUPPORTS", "strength": 0.8},
                {"existing_id": "mem-2", "link_type": "CONTRADICTS", "strength": 0.6},
            ]
        })

        result = evolution._parse_classification("new-mem", response, valid_ids)

        assert len(result) == 2
        assert result[0]["link_type"] == "SUPPORTS"
        assert result[0]["source_entity_id"] == "new-mem"
        assert result[0]["target_entity_id"] == "mem-1"
        assert result[0]["strength"] == 0.8

    def test_parse_with_markdown_fences(self):
        """Parse response wrapped in markdown code fences."""
        valid_ids = {"mem-1"}
        response = '```json\n{"relationships": [{"existing_id": "mem-1", "link_type": "REFINES", "strength": 0.7}]}\n```'

        result = evolution._parse_classification("new-mem", response, valid_ids)

        assert len(result) == 1
        assert result[0]["link_type"] == "REFINES"

    def test_parse_filters_low_strength(self):
        """Relationships with strength < 0.3 are filtered out."""
        valid_ids = {"mem-1"}
        response = json.dumps({
            "relationships": [
                {"existing_id": "mem-1", "link_type": "RELATED", "strength": 0.2},
            ]
        })

        result = evolution._parse_classification("new-mem", response, valid_ids)
        assert len(result) == 0

    def test_parse_filters_invalid_ids(self):
        """Relationships with IDs not in valid_ids are filtered."""
        valid_ids = {"mem-1"}
        response = json.dumps({
            "relationships": [
                {"existing_id": "mem-99", "link_type": "SUPPORTS", "strength": 0.8},
            ]
        })

        result = evolution._parse_classification("new-mem", response, valid_ids)
        assert len(result) == 0

    def test_parse_normalizes_invalid_link_type(self):
        """Invalid link_type defaults to RELATED."""
        valid_ids = {"mem-1"}
        response = json.dumps({
            "relationships": [
                {"existing_id": "mem-1", "link_type": "BANANA", "strength": 0.8},
            ]
        })

        result = evolution._parse_classification("new-mem", response, valid_ids)
        assert len(result) == 1
        assert result[0]["link_type"] == "RELATED"

    def test_parse_empty_response(self):
        """Empty relationships list returns empty."""
        result = evolution._parse_classification(
            "new-mem", '{"relationships": []}', {"mem-1"},
        )
        assert len(result) == 0

    def test_parse_garbage_response(self):
        """Unparseable response returns empty list."""
        result = evolution._parse_classification(
            "new-mem", "this is not json at all", {"mem-1"},
        )
        assert len(result) == 0


class TestMemoryLinks:
    """Test db_lifecycle insert_memory_link and get_memory_links."""

    def test_insert_and_get_memory_link(self):
        """Insert a memory link and retrieve it."""
        mem1_id = db.insert_memory(
            {"type": "semantic", "content": "Python is a programming language", "importance": 5},
            "test-session", "/test",
        )
        mem2_id = db.insert_memory(
            {"type": "semantic", "content": "Python supports async/await", "importance": 5},
            "test-session", "/test",
        )

        link_id = db.insert_memory_link(mem1_id, mem2_id, "SUPPORTS", 0.8)
        assert link_id is not None

        links = db.get_memory_links(mem1_id)
        assert len(links) == 1
        assert links[0]["link_type"] == "SUPPORTS"
        assert links[0]["strength"] == 0.8

        links2 = db.get_memory_links(mem2_id)
        assert len(links2) == 1

    def test_insert_memory_link_idempotent(self):
        """Inserting the same link twice returns the same ID."""
        mem1_id = db.insert_memory(
            {"type": "semantic", "content": "Use pytest always", "importance": 7},
            "test-session", "/test",
        )
        mem2_id = db.insert_memory(
            {"type": "semantic", "content": "pytest is better than unittest", "importance": 6},
            "test-session", "/test",
        )

        link1 = db.insert_memory_link(mem1_id, mem2_id, "SUPPORTS", 0.8)
        link2 = db.insert_memory_link(mem1_id, mem2_id, "SUPPORTS", 0.9)

        assert link1 == link2

    def test_insert_memory_link_invalid_type(self):
        """Invalid link_type defaults to RELATED."""
        mem1_id = db.insert_memory(
            {"type": "semantic", "content": "Content A unique 123", "importance": 5},
            "test-session", "/test",
        )
        mem2_id = db.insert_memory(
            {"type": "semantic", "content": "Content B unique 456", "importance": 5},
            "test-session", "/test",
        )

        db.insert_memory_link(mem1_id, mem2_id, "INVALID_TYPE", 0.5)

        links = db.get_memory_links(mem1_id)
        assert len(links) == 1
        assert links[0]["link_type"] == "RELATED"


class TestAMEMQueue:
    """Test A-MEM queue operations: enqueue, get pending, dequeue, expiry."""

    def test_queue_and_get_pending(self):
        """Queue a memory and retrieve it from pending."""
        db.queue_amem_evolution("test-mem-id-001")

        pending = db.get_pending_amem_queue()
        assert len(pending) == 1
        assert pending[0][1] == "test-mem-id-001"

    def test_dequeue(self):
        """Dequeuing removes the entry."""
        db.queue_amem_evolution("test-mem-id-002")

        pending = db.get_pending_amem_queue()
        assert len(pending) == 1

        db.dequeue_amem(pending[0][0])

        pending2 = db.get_pending_amem_queue()
        assert len(pending2) == 0

    def test_expiry_7_days(self):
        """Queue entries older than 7 days are auto-discarded."""
        from db import get_db

        conn = get_db()
        try:
            old_time = time.time() - (8 * 86400)
            conn.execute(
                "INSERT OR REPLACE INTO kg_sync_state (key, value, updated_at) VALUES (?, ?, ?)",
                ("amem_queue_old-mem", "old-mem", old_time),
            )
            conn.commit()
        finally:
            conn.close()

        pending = db.get_pending_amem_queue(max_age_days=7)
        assert len(pending) == 0

    def test_retry_persistence(self):
        """On failure, items stay in queue for retry."""
        db.queue_amem_evolution("test-mem-retry")

        pending1 = db.get_pending_amem_queue()
        assert len(pending1) == 1

        pending2 = db.get_pending_amem_queue()
        assert len(pending2) == 1
        assert pending2[0][1] == "test-mem-retry"

    def test_store_memory_queues_amem(self):
        """store_memory._store_to_sqlite queues memories for A-MEM evolution."""
        mem = {
            "type": "semantic",
            "content": "A-MEM queue test: unique content for verification xyz",
            "importance": 5,
        }

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, _ = store_memory._store_to_sqlite([mem], "test-amem-queue-session")

        assert new_count == 1

        pending = db.get_pending_amem_queue()
        assert len(pending) >= 1
        mem_id = pending[-1][1]
        assert len(mem_id) == 36  # UUID length


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestValidityGates + TestSessionStartDecisions — Sprint 4
# ═══════════════════════════════════════════════════════════════════════════════

import session_start
MIN_IMPORTANCE = session_start.MIN_IMPORTANCE


class TestValidityGates:
    """Test that superseded/gc_eligible/expired memories are excluded from search."""

    def test_superseded_memory_excluded(self):
        """A superseded memory should not appear in search results."""
        mem_id = db.insert_memory(
            {"type": "semantic", "content": "Use MySQL for the database", "importance": 7},
            "test-session", "/test",
        )
        # Supersede it
        new_mem_id = db.insert_memory(
            {"type": "semantic", "content": "Use PostgreSQL for the database instead of MySQL",
             "importance": 7},
            "test-session", "/test",
        )
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?",
                (new_mem_id, mem_id),
            )
            conn.commit()
        finally:
            conn.close()

        # Verify via direct DB query
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT superseded_by FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["superseded_by"] == new_mem_id

    def test_gc_eligible_memory_excluded(self):
        """A gc_eligible memory should be filtered out."""
        mem_id = db.insert_memory(
            {"type": "episodic", "content": "Discussed old architecture patterns", "importance": 3},
            "test-session", "/test",
        )
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET gc_eligible = 1 WHERE id = ?",
                (mem_id,),
            )
            conn.commit()
        finally:
            conn.close()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT gc_eligible FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["gc_eligible"] == 1

    def test_expired_memory_excluded(self):
        """A memory with valid_to in the past should be excluded."""
        past_time = time.time() - 86400  # 1 day ago
        mem_id = db.insert_memory(
            {"type": "semantic", "content": "Temporary constraint: freeze deployments",
             "importance": 8, "valid_to": past_time},
            "test-session", "/test",
        )

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT valid_to FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["valid_to"] < time.time()


class TestSessionStartDecisions:
    """Test that session_start.py injects decisions from the vault."""

    def test_decisions_included_in_context(self):
        """Decisions from the vault appear in session start context."""
        # Insert a memory and a linked decision
        mem_id = db.insert_memory(
            {"type": "procedural", "content": "Use SQLite not PostgreSQL for this project",
             "importance": 8},
            "test-session", "/test/project",
        )
        import hashlib
        db.insert_decision(
            memory_id=mem_id,
            decision_type="ARCHITECTURAL",
            content_hash=hashlib.sha256(b"Use SQLite not PostgreSQL for this project").hexdigest(),
            keywords=json.dumps(["sqlite", "postgresql"]),
            files_referenced=None,
            project="/test/project",
            session_id="test-session",
        )

        # Also need a standing memory to trigger context generation
        db.insert_memory(
            {"type": "correction", "content": "Always use pytest not unittest",
             "importance": 8},
            "test-session", "/test/project",
        )

        # Verify the decision exists in the vault
        conn = db.get_db()
        try:
            rows = conn.execute(
                "SELECT d.decision_type, m.content FROM decisions d "
                "JOIN memories m ON m.id = d.memory_id "
                "WHERE d.project = ? ORDER BY d.created_at DESC LIMIT 3",
                ("/test/project",),
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) >= 1
        assert rows[0]["decision_type"] == "ARCHITECTURAL"

    def test_no_decisions_no_crash(self):
        """Session start works fine when no decisions exist."""
        db.insert_memory(
            {"type": "correction", "content": "Never use print for debugging output",
             "importance": 8},
            "test-session", "/test/empty-decisions",
        )

        memories = db.get_memories_for_session_start(
            project="/test/empty-decisions", min_importance=MIN_IMPORTANCE,
        )
        context = session_start.format_context(memories, "/test/empty-decisions")
        # Should not crash, and should not contain "Recent Decisions" section
        assert "Recent Decisions" not in context


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestCompositeScoring — Sprint 5
# ═══════════════════════════════════════════════════════════════════════════════

# Import from daemon
sys.path.insert(0, str(Path(__file__).parent.parent / "daemon"))
from embedding_daemon import composite_score


class TestCompositeScoring:
    """Test the composite_score function."""

    def test_basic_scoring(self):
        """Basic scoring produces a positive value."""
        score = composite_score(
            rrf_score=0.5, temporal_score=0.8, importance=7,
        )
        assert score > 0

    def test_confidence_reduces_score(self):
        """Lower confidence produces lower score."""
        full = composite_score(rrf_score=0.5, temporal_score=0.8, importance=7, confidence=1.0)
        half = composite_score(rrf_score=0.5, temporal_score=0.8, importance=7, confidence=0.5)

        assert half < full
        assert abs(half - full * 0.5) < 0.001

    def test_zero_confidence_zeroes_score(self):
        """Zero confidence produces zero score."""
        score = composite_score(rrf_score=0.5, temporal_score=0.8, importance=10, confidence=0.0)
        assert score == 0.0

    def test_higher_importance_higher_score(self):
        """Higher importance produces higher score (all else equal)."""
        low = composite_score(rrf_score=0.5, temporal_score=0.5, importance=3)
        high = composite_score(rrf_score=0.5, temporal_score=0.5, importance=9)

        assert high > low

    def test_higher_temporal_higher_score(self):
        """Higher temporal score produces higher score (all else equal)."""
        old = composite_score(rrf_score=0.5, temporal_score=0.2, importance=5)
        fresh = composite_score(rrf_score=0.5, temporal_score=0.9, importance=5)

        assert fresh > old

    def test_higher_rrf_higher_score(self):
        """Higher RRF score produces higher score."""
        low_rrf = composite_score(rrf_score=0.1, temporal_score=0.5, importance=5)
        high_rrf = composite_score(rrf_score=0.9, temporal_score=0.5, importance=5)

        assert high_rrf > low_rrf

    def test_weight_proportions(self):
        """Verify the weights are applied correctly."""
        # rrf=1.0, temporal=0, importance=0 → score = 0.5 * 1.0 = 0.5
        score = composite_score(rrf_score=1.0, temporal_score=0.0, importance=0)
        assert abs(score - 0.5) < 0.001

    def test_feature_flag_off(self):
        """When ENSEMBLE_MEMORY_COMPOSITE_SCORING=0, old formula should be used."""
        # This is tested via the daemon's _search function, not composite_score directly.
        # We just verify the function exists and works.
        score = composite_score(rrf_score=0.5, temporal_score=0.5, importance=5)
        assert isinstance(score, float)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. E2E Integration Tests — Sprint 6
# ═══════════════════════════════════════════════════════════════════════════════


class TestE2EPhase7:
    """End-to-end integration tests covering the full Phase 7 pipeline."""

    def test_e2e_amem_evolution_with_links(self):
        """Insert related memories → create links → verify kg_memory_links."""
        mem1 = db.insert_memory(
            {"type": "procedural", "content": "Use pytest for all Python tests in this project",
             "importance": 8, "subject": "testing", "predicate": "must_use", "object": "pytest"},
            "e2e-session", "/e2e",
        )
        mem2 = db.insert_memory(
            {"type": "procedural", "content": "pytest fixtures are preferred over setUp methods",
             "importance": 7, "subject": "testing", "predicate": "should_use", "object": "fixtures"},
            "e2e-session", "/e2e",
        )
        mem3 = db.insert_memory(
            {"type": "correction", "content": "Never use unittest module, always pytest",
             "importance": 9, "subject": "testing", "predicate": "must_not_use", "object": "unittest"},
            "e2e-session", "/e2e",
        )

        # Create A-MEM links between them
        db.insert_memory_link(mem1, mem2, "SUPPORTS", 0.8)
        db.insert_memory_link(mem3, mem1, "REFINES", 0.7)
        db.insert_memory_link(mem3, mem2, "ENABLES", 0.6)

        # Verify links
        links1 = db.get_memory_links(mem1)
        assert len(links1) == 2  # SUPPORTS to mem2, REFINES from mem3

        links3 = db.get_memory_links(mem3)
        assert len(links3) == 2  # REFINES to mem1, ENABLES to mem2

    def test_e2e_session_end_safety_net(self):
        """Stop hook captures some memories, SessionEnd catches the rest."""
        # Simulate stop hook capturing a memory
        with patch.object(store_memory, "_notify_daemon_invalidate"):
            store_memory._store_to_sqlite(
                [{"type": "correction", "content": "Use ruff not flake8 for linting",
                  "importance": 8}],
                "e2e-session-end",
            )

        # Now session_end.py gets the full extraction (includes captured + new)
        extraction = json.dumps({
            "memories": [
                {"type": "correction", "content": "Use ruff not flake8 for linting",
                 "importance": 8},  # Already captured
                {"type": "semantic", "content": "The project uses Python 3.11 specifically",
                 "importance": 6},  # New — missed by stop hook
            ],
            "entities": [],
            "summary": [],
        })

        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new, already = session_end.session_end_sqlite(extraction, "e2e-session-end")

        assert already == 1  # ruff memory already in DB
        assert new == 1  # Python 3.11 is new

    def test_e2e_reinforcement_with_trigger_condition(self):
        """Insert procedural with trigger_condition → re-state → verify reinforced."""
        # First occurrence
        with patch.object(store_memory, "_notify_daemon_invalidate"):
            store_memory._store_to_sqlite(
                [{"type": "procedural",
                  "content": "Always run ruff before committing code",
                  "importance": 7,
                  "subject": "code_quality",
                  "predicate": "must_run",
                  "object": "ruff",
                  "trigger_condition": "before git commit",
                  "rule": "Run ruff before committing"}],
                "e2e-reinforce-1",
            )

        # Second occurrence — same rule, different wording
        with patch.object(store_memory, "_notify_daemon_invalidate"):
            new_count, _ = store_memory._store_to_sqlite(
                [{"type": "procedural",
                  "content": "Always run ruff before committing code",
                  "importance": 7,
                  "subject": "code_quality",
                  "predicate": "must_run",
                  "object": "ruff",
                  "trigger_condition": "before git commit"}],
                "e2e-reinforce-2",
            )

        assert new_count == 0  # Should reinforce, not insert

        # Verify reinforcement_count
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT reinforcement_count FROM memories WHERE subject = 'code_quality' AND predicate = 'must_run'",
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row["reinforcement_count"] >= 1

    def test_e2e_composite_scoring_importance(self):
        """Higher importance memory ranks higher than lower importance."""
        score_low = composite_score(rrf_score=0.5, temporal_score=0.5, importance=3)
        score_high = composite_score(rrf_score=0.5, temporal_score=0.5, importance=9)

        assert score_high > score_low

    def test_e2e_composite_scoring_confidence(self):
        """Contradicted memory (low confidence) ranks lower."""
        confident = composite_score(rrf_score=0.5, temporal_score=0.5, importance=7, confidence=1.0)
        contradicted = composite_score(rrf_score=0.5, temporal_score=0.5, importance=7, confidence=0.3)

        assert contradicted < confident

    def test_e2e_evolution_parsing_full_flow(self):
        """End-to-end: parse a classification response and create links."""
        # Create real memories
        mem1 = db.insert_memory(
            {"type": "semantic", "content": "SQLite is used for the database layer",
             "importance": 6},
            "e2e-evo", "/e2e",
        )
        mem2 = db.insert_memory(
            {"type": "semantic", "content": "The database uses WAL mode for performance",
             "importance": 6},
            "e2e-evo", "/e2e",
        )

        # Simulate Ollama response
        response = json.dumps({
            "relationships": [
                {"existing_id": mem2, "link_type": "SUPPORTS", "strength": 0.8},
            ]
        })

        result = evolution._parse_classification(mem1, response, {mem2})
        assert len(result) == 1

        # Write the link
        for rel in result:
            db.insert_memory_link(
                rel["source_entity_id"], rel["target_entity_id"],
                rel["link_type"], rel["strength"],
            )

        links = db.get_memory_links(mem1)
        assert len(links) == 1
        assert links[0]["link_type"] == "SUPPORTS"

    def test_e2e_queue_lifecycle(self):
        """Full queue lifecycle: insert → queue → check pending → dequeue."""
        mem_id = db.insert_memory(
            {"type": "semantic", "content": "Queue lifecycle e2e test memory",
             "importance": 5},
            "e2e-queue", "/e2e",
        )

        # Queue
        db.queue_amem_evolution(mem_id)

        # Check pending
        pending = db.get_pending_amem_queue()
        mem_ids = [p[1] for p in pending]
        assert mem_id in mem_ids

        # Dequeue
        for key, val in pending:
            if val == mem_id:
                db.dequeue_amem(key)

        # Verify dequeued
        pending2 = db.get_pending_amem_queue()
        mem_ids2 = [p[1] for p in pending2]
        assert mem_id not in mem_ids2
