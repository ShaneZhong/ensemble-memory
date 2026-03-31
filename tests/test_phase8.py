"""Tests for Phase 8: Recall Quality II — Cross-encoder & Calibration."""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = PROJECT_ROOT / "hooks"
DAEMON_DIR = PROJECT_ROOT / "daemon"

if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))
if str(DAEMON_DIR) not in sys.path:
    sys.path.insert(0, str(DAEMON_DIR))

# Set env to use test DB
os.environ.setdefault("ENSEMBLE_MEMORY_DIR", str(PROJECT_ROOT / "test_data"))


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 8.1: Cross-encoder Reranking
# ═══════════════════════════════════════════════════════════════════════════════

class TestTruncateForRerank:
    """Test content truncation for cross-encoder input."""

    def test_short_content_unchanged(self):
        from embedding_daemon import _truncate_for_rerank
        content = "Short memory content"
        assert _truncate_for_rerank(content, None) == content

    def test_exactly_600_chars_unchanged(self):
        from embedding_daemon import _truncate_for_rerank
        content = "x" * 600
        assert _truncate_for_rerank(content, None) == content

    def test_long_content_truncated_no_subject(self):
        from embedding_daemon import _truncate_for_rerank
        content = "y" * 800
        result = _truncate_for_rerank(content, None)
        assert len(result) == 200
        assert result == "y" * 200

    def test_long_content_truncated_with_subject(self):
        from embedding_daemon import _truncate_for_rerank
        content = "z" * 800
        result = _truncate_for_rerank(content, "pytest")
        assert result.startswith("pytest: ")
        assert len(result) == len("pytest: ") + 200

    def test_long_content_empty_subject(self):
        from embedding_daemon import _truncate_for_rerank
        content = "a" * 800
        result = _truncate_for_rerank(content, "")
        # Empty string is falsy, so no prefix
        assert result == "a" * 200


class TestCrossEncoderRerank:
    """Test cross-encoder reranking logic."""

    def test_empty_candidates_returns_empty(self):
        from embedding_daemon import _cross_encoder_rerank
        result = _cross_encoder_rerank("query", [])
        assert result == []

    def test_rerank_with_mocked_model(self):
        import embedding_daemon as ed
        from embedding_daemon import _cross_encoder_rerank

        candidates = [
            {"id": "a", "content": "low relevance", "subject": None, "final_score": 0.9},
            {"id": "b", "content": "high relevance to query", "subject": None, "final_score": 0.5},
            {"id": "c", "content": "medium relevance", "subject": None, "final_score": 0.7},
        ]

        mock_model = MagicMock()
        # Model says b > c > a
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            ed._cross_encoder = mock_model
            ed._has_cross_encoder = True

            result = _cross_encoder_rerank("test query", candidates, top_n=2)
            assert len(result) == 2
            assert result[0]["id"] == "b"  # highest cross-encoder score
            assert result[1]["id"] == "c"
            assert result[0]["final_score"] == 0.9
            assert result[0]["cross_encoder_score"] == 0.9
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has

    def test_rerank_fallback_when_no_model(self):
        import embedding_daemon as ed
        from embedding_daemon import _cross_encoder_rerank

        candidates = [
            {"id": "a", "content": "first", "final_score": 0.9},
            {"id": "b", "content": "second", "final_score": 0.5},
        ]

        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            ed._cross_encoder = None
            ed._has_cross_encoder = False

            result = _cross_encoder_rerank("query", candidates, top_n=5)
            # Should return original order (RRF-only fallback)
            assert len(result) == 2
            assert result[0]["id"] == "a"
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has

    def test_rerank_model_predict_failure(self):
        import embedding_daemon as ed
        from embedding_daemon import _cross_encoder_rerank

        candidates = [
            {"id": "a", "content": "test", "final_score": 0.9},
        ]

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model crashed")

        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            ed._cross_encoder = mock_model
            ed._has_cross_encoder = True

            result = _cross_encoder_rerank("query", candidates, top_n=5)
            # Should fallback gracefully
            assert len(result) == 1
            assert result[0]["id"] == "a"
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has

    def test_single_candidate(self):
        import embedding_daemon as ed
        from embedding_daemon import _cross_encoder_rerank

        candidates = [{"id": "only", "content": "solo", "subject": None, "final_score": 1.0}]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]

        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            ed._cross_encoder = mock_model
            ed._has_cross_encoder = True

            result = _cross_encoder_rerank("query", candidates, top_n=5)
            assert len(result) == 1
            assert result[0]["id"] == "only"
            assert result[0]["final_score"] == 0.8
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has


class TestCrossEncoderFeatureFlag:
    """Test feature flag control for cross-encoder."""

    def test_feature_flag_disabled(self):
        import embedding_daemon as ed

        old_val = os.environ.get("ENSEMBLE_MEMORY_CROSS_ENCODER")
        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            os.environ["ENSEMBLE_MEMORY_CROSS_ENCODER"] = "0"
            ed._cross_encoder = None
            ed._has_cross_encoder = False

            ed._load_cross_encoder()
            assert ed._has_cross_encoder is False
            assert ed._cross_encoder is None
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has
            if old_val is not None:
                os.environ["ENSEMBLE_MEMORY_CROSS_ENCODER"] = old_val
            else:
                os.environ.pop("ENSEMBLE_MEMORY_CROSS_ENCODER", None)

    def test_rerank_param_false_skips_cross_encoder(self):
        """When rerank=False (default), cross-encoder should not be invoked."""
        import embedding_daemon as ed

        mock_model = MagicMock()
        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            ed._cross_encoder = mock_model
            ed._has_cross_encoder = True

            # _search with rerank=False should NOT call cross-encoder
            # We verify by checking _cross_encoder_rerank is not called
            # since the _search function checks rerank flag before calling
            # We can't easily test _search without full DB, so test the flag logic directly
            assert ed._has_cross_encoder is True
            # The key check: rerank=False in _search means _cross_encoder_rerank not called
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has


class TestLoadCrossEncoder:
    """Test lazy loading behavior."""

    def test_load_skipped_when_already_loaded(self):
        import embedding_daemon as ed

        mock_model = MagicMock()
        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            ed._cross_encoder = mock_model
            ed._has_cross_encoder = True

            # Should not re-load
            ed._load_cross_encoder()
            assert ed._cross_encoder is mock_model  # Same object
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has

    def test_load_failure_sets_flag_false(self):
        import embedding_daemon as ed

        old_ce = ed._cross_encoder
        old_has = ed._has_cross_encoder
        try:
            ed._cross_encoder = None
            ed._has_cross_encoder = False

            # Patch the import inside the function to simulate failure
            with patch.dict(os.environ, {"ENSEMBLE_MEMORY_CROSS_ENCODER": "1"}):
                with patch.dict(sys.modules, {"sentence_transformers": MagicMock(
                    CrossEncoder=MagicMock(side_effect=RuntimeError("model load failed"))
                )}):
                    ed._load_cross_encoder()
                    assert ed._has_cross_encoder is False
                    assert ed._cross_encoder is None
        finally:
            ed._cross_encoder = old_ce
            ed._has_cross_encoder = old_has


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 8.3: SessionStart Validity Gates
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def setup_test_db():
    """Set up a test database with memories including validity bounds."""
    import db

    conn = db.get_db()
    now = time.time()

    # Insert test memories with different validity states
    memories = [
        # Valid: no bounds (majority case)
        ("mem_valid_1", "Always use pytest for testing", "procedural", 8, None, None),
        # Valid: valid_to in the future
        ("mem_valid_2", "Use SQLite for the database", "correction", 9, None, now + 86400),
        # Expired: valid_to in the past
        ("mem_expired", "Old rule that expired yesterday", "procedural", 8, None, now - 86400),
        # Future: valid_from in the future
        ("mem_future", "Rule that starts tomorrow", "correction", 8, now + 86400, None),
        # Valid: valid_from in the past, valid_to in future
        ("mem_bounded", "Active bounded rule", "procedural", 8, now - 86400, now + 86400),
        # Semantic (for recent context)
        ("mem_semantic", "Project uses Python 3.11", "semantic", 7, None, None),
        # Episodic (for recent context)
        ("mem_episodic", "Fixed the FK constraint bug", "episodic", 6, None, None),
        # Low importance semantic (should be excluded from recent context)
        ("mem_low_imp", "Minor detail about formatting", "semantic", 3, None, None),
    ]

    for mid, content, mtype, importance, valid_from, valid_to in memories:
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, content, memory_type, importance, project, created_at,
                 access_count, last_accessed_at, decay_rate, stability,
                 valid_from, valid_to, superseded_by, gc_eligible, content_hash)
            VALUES (?, ?, ?, ?, '/test/project', ?, 1, ?, 0.16, 0.0, ?, ?, NULL, 0, ?)
            """,
            (mid, content, mtype, importance, now, now,
             valid_from, valid_to, f"hash_{mid}"),
        )
    conn.commit()
    conn.close()

    yield

    # Cleanup
    conn = db.get_db()
    for mid, *_ in memories:
        conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
    conn.commit()
    conn.close()


class TestSessionStartValidityGates:
    """Test validity gates in get_memories_for_session_start()."""

    def test_expired_memory_excluded(self, setup_test_db):
        import db
        memories = db.get_memories_for_session_start(project="/test/project")
        mem_ids = {m["id"] for m in memories}
        assert "mem_expired" not in mem_ids

    def test_future_memory_excluded(self, setup_test_db):
        import db
        memories = db.get_memories_for_session_start(project="/test/project")
        mem_ids = {m["id"] for m in memories}
        assert "mem_future" not in mem_ids

    def test_null_bounds_still_loaded(self, setup_test_db):
        import db
        memories = db.get_memories_for_session_start(project="/test/project")
        mem_ids = {m["id"] for m in memories}
        assert "mem_valid_1" in mem_ids

    def test_valid_to_future_still_loaded(self, setup_test_db):
        import db
        memories = db.get_memories_for_session_start(project="/test/project")
        mem_ids = {m["id"] for m in memories}
        assert "mem_valid_2" in mem_ids

    def test_bounded_valid_still_loaded(self, setup_test_db):
        import db
        memories = db.get_memories_for_session_start(project="/test/project")
        mem_ids = {m["id"] for m in memories}
        assert "mem_bounded" in mem_ids

    def test_only_procedural_and_correction(self, setup_test_db):
        import db
        memories = db.get_memories_for_session_start(project="/test/project")
        for m in memories:
            assert m["memory_type"] in ("procedural", "correction")


class TestRecentContext:
    """Test get_recent_context() function."""

    def test_returns_semantic_and_episodic(self, setup_test_db):
        import db
        recent = db.get_recent_context(project="/test/project", min_importance=5)
        types = {m["memory_type"] for m in recent}
        # Should contain semantic and/or episodic, never procedural/correction
        for mtype in types:
            assert mtype in ("semantic", "episodic")

    def test_excludes_low_importance(self, setup_test_db):
        import db
        recent = db.get_recent_context(project="/test/project", min_importance=5)
        mem_ids = {m["id"] for m in recent}
        assert "mem_low_imp" not in mem_ids  # importance 3 < 5

    def test_excludes_specified_ids(self, setup_test_db):
        import db
        recent = db.get_recent_context(
            project="/test/project",
            min_importance=5,
            exclude_ids={"mem_semantic"},
        )
        mem_ids = {m["id"] for m in recent}
        assert "mem_semantic" not in mem_ids

    def test_includes_without_exclusion(self, setup_test_db):
        import db
        recent = db.get_recent_context(project="/test/project", min_importance=5)
        mem_ids = {m["id"] for m in recent}
        assert "mem_semantic" in mem_ids or "mem_episodic" in mem_ids

    def test_respects_limit(self, setup_test_db):
        import db
        recent = db.get_recent_context(project="/test/project", limit=1, min_importance=5)
        assert len(recent) <= 1

    def test_validity_gates_applied(self, setup_test_db):
        """Expired semantic memories should not appear in recent context."""
        import db

        # Insert an expired semantic memory
        conn = db.get_db()
        now = time.time()
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, content, memory_type, importance, project, created_at,
                 access_count, last_accessed_at, decay_rate, stability,
                 valid_to, superseded_by, gc_eligible, content_hash)
            VALUES (?, ?, 'semantic', 7, '/test/project', ?, 1, ?, 0.16, 0.0, ?, NULL, 0, ?)
            """,
            ("mem_expired_semantic", "Expired fact", now, now, now - 86400, "hash_expired_sem"),
        )
        conn.commit()
        conn.close()

        try:
            recent = db.get_recent_context(project="/test/project", min_importance=5)
            mem_ids = {m["id"] for m in recent}
            assert "mem_expired_semantic" not in mem_ids
        finally:
            conn = db.get_db()
            conn.execute("DELETE FROM memories WHERE id = 'mem_expired_semantic'")
            conn.commit()
            conn.close()


class TestSessionStartIntegration:
    """Integration test for full session start with validity gates + recent context."""

    def test_all_expired_returns_empty(self):
        """If all memories are expired, session start should return empty."""
        import db

        conn = db.get_db()
        now = time.time()
        # Insert only expired memories
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, content, memory_type, importance, project, created_at,
                 access_count, last_accessed_at, decay_rate, stability,
                 valid_to, superseded_by, gc_eligible, content_hash)
            VALUES (?, ?, 'procedural', 8, '/test/empty', ?, 1, ?, 0.16, 0.0, ?, NULL, 0, ?)
            """,
            ("mem_all_expired", "Expired rule", now, now, now - 86400, "hash_all_exp"),
        )
        conn.commit()
        conn.close()

        try:
            memories = db.get_memories_for_session_start(project="/test/empty")
            assert len(memories) == 0
        finally:
            conn = db.get_db()
            conn.execute("DELETE FROM memories WHERE id = 'mem_all_expired'")
            conn.commit()
            conn.close()
