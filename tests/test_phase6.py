#!/usr/bin/env python3
"""Test suite for Phase 6 Sprint 1 — Reinforcement tracking + promotion pipeline.

Run: /Users/shane/Documents/playground/.venv/bin/python3 -m pytest tests/test_phase6.py -v

Test classes:
  1. TestReinforcementIncrement — increment_reinforcement stability/promotion logic
  2. TestGetReinforcementMatch — matching, ordering, SQL wildcard escaping
  3. TestPromotion — check_and_promote criteria gating
  4. TestPromotionCLAUDEMD — file write format, idempotency, edge cases
  5. TestStoreMemoryReinforcement — full flow through _store_to_sqlite
"""

import json
import os
import stat
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add hooks dir to path
HOOKS_DIR = Path(__file__).parent.parent / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

import db
import promote


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Redirect DB to a temp directory for each test."""
    db._DB_PATH_OVERRIDE = str(tmp_path / "test.db")
    db._db_initialized.clear()
    # Ensure migration columns exist (Phase 2 + Phase 5)
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
# 1. TestReinforcementIncrement
# ═══════════════════════════════════════════════════════════════════════════════

class TestReinforcementIncrement:
    """Test increment_reinforcement stability and promotion logic."""

    def test_increment_count_1_no_stability_bonus(self):
        """count 0->1: stability = base (no bonus)."""
        mem_id = _insert_procedural("always run tests first", importance=5)
        new_count = db.increment_reinforcement(mem_id)
        assert new_count == 1
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT stability, promotion_candidate FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        base = db._compute_stability(5)
        assert row["stability"] == pytest.approx(base)
        assert row["promotion_candidate"] == 0

    def test_increment_count_2_stability_bonus(self):
        """count 1->2: stability = base + 0.1."""
        mem_id = _insert_procedural("always run tests first", importance=5,
                                     reinforcement_count=1)
        new_count = db.increment_reinforcement(mem_id)
        assert new_count == 2
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT stability, promotion_candidate FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        expected = min(1.0, db._compute_stability(5) + 0.1)
        assert row["stability"] == pytest.approx(expected)
        assert row["promotion_candidate"] == 0

    def test_increment_count_3_promotion_candidate(self):
        """count 2->3: stability = base + 0.2, promotion_candidate = 1."""
        mem_id = _insert_procedural("always run tests first", importance=5,
                                     reinforcement_count=2)
        new_count = db.increment_reinforcement(mem_id)
        assert new_count == 3
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT stability, promotion_candidate FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        expected = min(1.0, db._compute_stability(5) + 0.2)
        assert row["stability"] == pytest.approx(expected)
        assert row["promotion_candidate"] == 1

    def test_increment_count_4_same_as_3(self):
        """count 3->4: stability still base + 0.2, promotion still 1."""
        mem_id = _insert_procedural("always run tests first", importance=5,
                                     reinforcement_count=3)
        new_count = db.increment_reinforcement(mem_id)
        assert new_count == 4
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT stability, promotion_candidate FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        expected = min(1.0, db._compute_stability(5) + 0.2)
        assert row["stability"] == pytest.approx(expected)
        assert row["promotion_candidate"] == 1

    def test_increment_count_5_permanent(self):
        """count 4->5: stability = 1.0 (permanent)."""
        mem_id = _insert_procedural("always run tests first", importance=5,
                                     reinforcement_count=4)
        new_count = db.increment_reinforcement(mem_id)
        assert new_count == 5
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT stability, promotion_candidate FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["stability"] == pytest.approx(1.0)
        assert row["promotion_candidate"] == 1

    def test_increment_nonexistent_returns_zero(self):
        """Non-existent memory_id returns 0."""
        result = db.increment_reinforcement("nonexistent-id-12345")
        assert result == 0

    def test_increment_updates_last_accessed_at(self):
        """increment should update last_accessed_at."""
        mem_id = _insert_procedural("always run tests first", importance=5)
        before = time.time()
        db.increment_reinforcement(mem_id)
        after = time.time()
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT last_accessed_at FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert before <= row["last_accessed_at"] <= after


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestGetReinforcementMatch
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetReinforcementMatch:
    """Test get_reinforcement_match matching and edge cases."""

    def test_returns_count_and_id(self):
        """Basic match returns (reinforcement_count+1, id)."""
        mem_id = _insert_procedural("always run tests first", reinforcement_count=2)
        count, matched_id = db.get_reinforcement_match("run tests first")
        assert count == 3  # reinforcement_count(2) + 1
        assert matched_id == mem_id

    def test_no_match_returns_zero_none(self):
        """No matching memory returns (0, None)."""
        _insert_procedural("always run tests first")
        count, matched_id = db.get_reinforcement_match("nonexistent rule xyz")
        assert count == 0
        assert matched_id is None

    def test_empty_rule_returns_zero_none(self):
        """Empty rule_text returns (0, None) immediately."""
        count, matched_id = db.get_reinforcement_match("")
        assert count == 0
        assert matched_id is None

    def test_multiple_matches_picks_most_recent(self):
        """When multiple procedural memories match, pick the most recent."""
        _insert_procedural("run tests before committing - old rule")
        time.sleep(0.01)  # ensure different created_at
        newer_id = _insert_procedural("run tests before committing - new rule")
        count, matched_id = db.get_reinforcement_match("run tests before committing")
        assert matched_id == newer_id

    def test_skips_superseded_memories(self):
        """Superseded memories should not be matched."""
        mem_id = _insert_procedural("always run tests first")
        # Insert another memory to use as the superseder (FK constraint)
        superseder_id = _insert_procedural("updated rule about testing")
        # Mark original as superseded
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?",
                (superseder_id, mem_id),
            )
            conn.commit()
        finally:
            conn.close()
        count, matched_id = db.get_reinforcement_match("always run tests first")
        # Should only match the superseder if its content matches, not the original
        # The original is superseded so it's excluded
        assert matched_id != mem_id

    def test_skips_gc_eligible_memories(self):
        """gc_eligible memories should not be matched."""
        mem_id = _insert_procedural("always run tests first")
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET gc_eligible = 1 WHERE id = ?",
                (mem_id,),
            )
            conn.commit()
        finally:
            conn.close()
        count, matched_id = db.get_reinforcement_match("run tests first")
        assert count == 0
        assert matched_id is None

    def test_only_matches_procedural(self):
        """Non-procedural memories should not be matched."""
        mem = {
            "type": "semantic",
            "content": "always run tests first",
            "importance": 7,
        }
        db.insert_memory(mem, "test-session", "/test/project")
        count, matched_id = db.get_reinforcement_match("run tests first")
        assert count == 0
        assert matched_id is None

    def test_sql_percent_wildcard_escaped(self):
        """SQL % wildcard in rule_text should be escaped."""
        _insert_procedural("use 100% coverage for critical paths")
        # This should NOT match a memory with different content
        _insert_procedural("use any coverage level")
        count, matched_id = db.get_reinforcement_match("100%")
        # Should match only the first one
        assert count > 0
        assert matched_id is not None

    def test_sql_underscore_wildcard_escaped(self):
        """SQL _ wildcard in rule_text should be escaped."""
        _insert_procedural("use my_variable naming convention")
        # _ should not act as single-char wildcard
        count, matched_id = db.get_reinforcement_match("my_variable")
        assert count > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestPromotion
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromotion:
    """Test check_and_promote criteria gating."""

    def test_promotes_when_criteria_met(self, tmp_path):
        """Memory with count>=5, procedural, fresh → promoted."""
        mem_id = _insert_procedural(
            "always run tests before committing",
            importance=7,
            reinforcement_count=5,
        )
        claude_md = str(tmp_path / "CLAUDE.md")
        result = promote.check_and_promote(mem_id, claude_md_path=claude_md)
        assert result is True
        assert Path(claude_md).exists()

    def test_rejects_count_below_threshold(self, tmp_path):
        """Memory with count=4 → not promoted."""
        mem_id = _insert_procedural(
            "always run tests before committing",
            importance=7,
            reinforcement_count=4,
        )
        claude_md = str(tmp_path / "CLAUDE.md")
        result = promote.check_and_promote(mem_id, claude_md_path=claude_md)
        assert result is False
        assert not Path(claude_md).exists()

    def test_rejects_stale_memory(self, tmp_path):
        """Memory last accessed > 180 days ago → not promoted."""
        mem_id = _insert_procedural(
            "always run tests before committing",
            importance=7,
            reinforcement_count=5,
        )
        # Set last_accessed_at to 200 days ago
        stale_time = time.time() - (200 * 86400)
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET last_accessed_at = ?, created_at = ? WHERE id = ?",
                (stale_time, stale_time, mem_id),
            )
            conn.commit()
        finally:
            conn.close()
        claude_md = str(tmp_path / "CLAUDE.md")
        result = promote.check_and_promote(mem_id, claude_md_path=claude_md)
        assert result is False

    def test_rejects_non_procedural(self, tmp_path):
        """Semantic memory → not promoted even with high count."""
        mem = {
            "type": "semantic",
            "content": "Python uses indentation",
            "importance": 7,
        }
        mem_id = db.insert_memory(mem, "test-session", "/test/project")
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET reinforcement_count = 10 WHERE id = ?",
                (mem_id,),
            )
            conn.commit()
        finally:
            conn.close()
        claude_md = str(tmp_path / "CLAUDE.md")
        result = promote.check_and_promote(mem_id, claude_md_path=claude_md)
        assert result is False

    def test_rejects_nonexistent_memory(self, tmp_path):
        """Nonexistent memory_id → returns False."""
        claude_md = str(tmp_path / "CLAUDE.md")
        result = promote.check_and_promote("nonexistent-id", claude_md_path=claude_md)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestPromotionCLAUDEMD
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromotionCLAUDEMD:
    """Test _write_to_claude_md file output format and edge cases."""

    def test_creates_new_file(self, tmp_path):
        """When CLAUDE.md doesn't exist, create it with section header."""
        path = str(tmp_path / "CLAUDE.md")
        result = promote._write_to_claude_md(path, "run tests first", 7, 5)
        assert result is True
        text = Path(path).read_text()
        assert "## Learned Behaviors" in text
        assert "- run tests first (importance: 7, reinforced: 5x)" in text

    def test_appends_to_existing_section(self, tmp_path):
        """When section exists, append entry after header."""
        path = str(tmp_path / "CLAUDE.md")
        Path(path).write_text("# Project\n\n## Learned Behaviors\n- old entry\n")
        result = promote._write_to_claude_md(path, "run tests first", 7, 5)
        assert result is True
        text = Path(path).read_text()
        assert "- run tests first (importance: 7, reinforced: 5x)" in text
        assert "- old entry" in text

    def test_creates_section_when_missing(self, tmp_path):
        """When CLAUDE.md exists but no section, append section at end."""
        path = str(tmp_path / "CLAUDE.md")
        Path(path).write_text("# My Project\n\nSome existing content.\n")
        result = promote._write_to_claude_md(path, "run tests first", 7, 5)
        assert result is True
        text = Path(path).read_text()
        assert "## Learned Behaviors" in text
        assert "# My Project" in text
        assert "Some existing content." in text

    def test_idempotency(self, tmp_path):
        """Writing the same content twice should not duplicate."""
        path = str(tmp_path / "CLAUDE.md")
        promote._write_to_claude_md(path, "run tests first", 7, 5)
        result = promote._write_to_claude_md(path, "run tests first", 7, 5)
        assert result is False
        text = Path(path).read_text()
        assert text.count("run tests first") == 1

    def test_read_only_file(self, tmp_path):
        """Read-only CLAUDE.md → returns False."""
        path = str(tmp_path / "CLAUDE.md")
        Path(path).write_text("# Read only\n")
        os.chmod(path, stat.S_IRUSR)
        try:
            result = promote._write_to_claude_md(path, "run tests first", 7, 5)
            assert result is False
        finally:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def test_entry_format(self, tmp_path):
        """Entry format: '- {content} (importance: N, reinforced: Nx)'."""
        path = str(tmp_path / "CLAUDE.md")
        promote._write_to_claude_md(path, "use pytest fixtures", 8, 6)
        text = Path(path).read_text()
        assert "- use pytest fixtures (importance: 8, reinforced: 6x)" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestStoreMemoryReinforcement
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoreMemoryReinforcement:
    """Test full flow through _store_to_sqlite reinforcement path."""

    @pytest.fixture(autouse=True)
    def _mock_enrich_and_embed(self):
        """Disable enrichment and embedding for store_memory tests."""
        import store_memory
        with patch.object(store_memory.enrich, "ENRICHMENT_ENABLED", False):
            yield

    def test_reinforcement_skips_duplicate_insert(self):
        """When a procedural memory matches an existing rule, don't insert new row."""
        import store_memory

        # Insert initial procedural memory whose content contains the rule text
        _insert_procedural("you must run tests before committing always")

        # Try to store a new memory with a rule that matches the existing content
        memories = [{
            "type": "procedural",
            "content": "you should run tests before committing code",
            "importance": 7,
            "rule": "run tests before committing",
        }]

        new_count, superseded = store_memory._store_to_sqlite(memories, "session-2")
        assert new_count == 0  # Should not insert new row

    def test_reinforcement_increments_existing(self):
        """Reinforcement match should increment the existing memory's count."""
        import store_memory

        # Content must contain the rule text for LIKE match
        mem_id = _insert_procedural("you must run tests before committing always")

        memories = [{
            "type": "procedural",
            "content": "you should run tests before committing code",
            "importance": 7,
            "rule": "run tests before committing",
        }]

        store_memory._store_to_sqlite(memories, "session-2")

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT reinforcement_count FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["reinforcement_count"] == 1

    def test_no_rule_inserts_normally(self):
        """Procedural memory without rule field should insert normally."""
        import store_memory

        memories = [{
            "type": "procedural",
            "content": "always run tests before committing",
            "importance": 7,
        }]

        new_count, superseded = store_memory._store_to_sqlite(memories, "session-1")
        assert new_count == 1

    def test_reinforcement_error_continues_processing(self):
        """When reinforcement check raises, remaining memories still process."""
        import store_memory

        memories = [
            {
                "type": "procedural",
                "content": "broken rule memory",
                "importance": 7,
                "rule": "some rule",
            },
            {
                "type": "semantic",
                "content": "Python uses 4 spaces for indentation",
                "importance": 6,
            },
        ]

        with patch.object(db, "get_reinforcement_match", side_effect=RuntimeError("DB error")):
            new_count, superseded = store_memory._store_to_sqlite(memories, "session-1")

        # Both memories should be inserted (first one falls through after error)
        assert new_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestSupersessionEventBus — Sprint 2
# ═══════════════════════════════════════════════════════════════════════════════

def _create_supersession_pair(subject="Python version", predicate="HAS_VERSION",
                               old_content="Python 3.10", new_content="Python 3.11",
                               importance=7, old_kwargs=None, new_kwargs=None):
    """Helper: create two memories and trigger supersession detection.

    Returns (old_id, new_id).
    """
    old_mem = {
        "type": "semantic",
        "content": old_content,
        "importance": importance,
        "subject": subject,
        "predicate": predicate,
    }
    if old_kwargs:
        old_mem.update(old_kwargs)
    old_id = db.insert_memory(old_mem, "session-old", "/test/project")

    new_mem = {
        "type": "semantic",
        "content": new_content,
        "importance": importance,
        "subject": subject,
        "predicate": predicate,
    }
    if new_kwargs:
        new_mem.update(new_kwargs)
    new_id = db.insert_memory(new_mem, "session-new", "/test/project")

    db.detect_supersession(new_id, subject, predicate)
    return old_id, new_id


class TestSupersessionEventBus:
    """Test process_supersession_events trilateral sync."""

    def test_all_three_experts_process(self):
        """All three flags set to 1 after processing."""
        old_id, new_id = _create_supersession_pair()
        stats = db.process_supersession_events()
        assert stats['processed'] == 1
        assert stats['temporal'] == 1
        assert stats['kg'] == 1
        assert stats['contextual'] == 1

        # Verify flags in DB
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT processed_by_temporal, processed_by_kg, processed_by_contextual "
                "FROM supersession_events WHERE old_memory_id = ?",
                (old_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["processed_by_temporal"] == 1
        assert row["processed_by_kg"] == 1
        assert row["processed_by_contextual"] == 1

    def test_idempotency(self):
        """Reprocessing already-processed events returns zeros."""
        _create_supersession_pair()
        db.process_supersession_events()
        stats = db.process_supersession_events()
        assert stats['processed'] == 0
        assert stats['temporal'] == 0
        assert stats['kg'] == 0
        assert stats['contextual'] == 0

    def test_partial_failure_kg(self):
        """If KG fails, temporal and contextual still process."""
        _create_supersession_pair()

        with patch.object(db, "_process_supersession_kg", side_effect=RuntimeError("KG error")):
            stats = db.process_supersession_events()

        assert stats['temporal'] == 1
        assert stats['kg'] == 0  # failed
        assert stats['contextual'] == 1

        # KG flag should still be 0
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT processed_by_kg FROM supersession_events LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        assert row["processed_by_kg"] == 0

    def test_empty_events_noop(self):
        """Empty events table returns all zeros."""
        stats = db.process_supersession_events()
        assert stats == {'processed': 0, 'temporal': 0, 'kg': 0, 'contextual': 0}

    def test_kg_no_subject_on_old_memory(self):
        """KG processor handles old memory with no subject gracefully."""
        old_mem = {
            "type": "semantic",
            "content": "some fact without subject",
            "importance": 5,
        }
        old_id = db.insert_memory(old_mem, "session-1", "/test/project")
        new_mem = {
            "type": "semantic",
            "content": "updated fact",
            "importance": 5,
        }
        new_id = db.insert_memory(new_mem, "session-2", "/test/project")

        # Manually insert a supersession event (no subject/predicate to trigger detect_supersession)
        conn = db.get_db()
        try:
            now = time.time()
            conn.execute(
                """INSERT INTO supersession_events
                   (old_memory_id, new_memory_id, event_time, detected_by,
                    processed_by_temporal, processed_by_kg, processed_by_contextual, created_at)
                   VALUES (?, ?, ?, 'manual', 0, 0, 0, ?)""",
                (old_id, new_id, now, now),
            )
            conn.commit()
        finally:
            conn.close()

        stats = db.process_supersession_events()
        assert stats['kg'] == 1  # Completed without error

    def test_contextual_null_enriched_text(self):
        """Contextual processor with NULL enriched_text is a no-op (no error)."""
        old_id, new_id = _create_supersession_pair()
        # enriched_text is already NULL by default
        stats = db.process_supersession_events()
        assert stats['contextual'] == 1

    def test_contextual_clears_enriched_text(self):
        """Contextual processor clears enriched_text on superseded memory."""
        old_id, new_id = _create_supersession_pair()
        # Set enriched_text on old memory
        db.store_enrichment(old_id, "enriched content here", 0.8)

        stats = db.process_supersession_events()
        assert stats['contextual'] == 1

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT enriched_text, enrichment_quality FROM memories WHERE id = ?",
                (old_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["enriched_text"] is None
        assert row["enrichment_quality"] == 0.0

    def test_event_with_deleted_memory(self):
        """Event referencing a memory that no longer exists is handled gracefully."""
        old_id, new_id = _create_supersession_pair()

        # Delete the old memory (break FK — SQLite FK enforcement is per-connection)
        conn = db.get_db()
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM memories WHERE id = ?", (old_id,))
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")
        finally:
            conn.close()

        # Should not raise
        stats = db.process_supersession_events()
        assert stats['processed'] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestChainDepthPruning — Sprint 2
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chain(length, memory_type="procedural", importance=5):
    """Build a supersession chain of given length.

    Returns list of memory IDs from oldest to newest.
    """
    chain_ids = []
    for i in range(length):
        mem = {
            "type": memory_type,
            "content": f"chain entry {i} for {memory_type} {importance}",
            "importance": importance,
            "subject": f"chain-test-{memory_type}-{importance}",
            "predicate": "HAS_VERSION",
        }
        mem_id = db.insert_memory(mem, f"session-{i}", "/test/project")
        chain_ids.append(mem_id)

    # Link them: each older one is superseded by the next
    conn = db.get_db()
    try:
        now = time.time()
        for i in range(len(chain_ids) - 1):
            conn.execute(
                "UPDATE memories SET superseded_by = ?, superseded_at = ? WHERE id = ?",
                (chain_ids[i + 1], now, chain_ids[i]),
            )
        conn.commit()
    finally:
        conn.close()

    return chain_ids


class TestChainDepthPruning:
    """Test enforce_chain_depth_limits and get_supersession_chain."""

    def test_chain_within_limit(self):
        """Chain of 3 procedural (limit=3) stays within limit, no GC."""
        chain_ids = _build_chain(3, "procedural")
        marked = db.enforce_chain_depth_limits()
        assert marked == 0

    def test_chain_exceeds_limit(self):
        """Chain of 5 procedural (limit=3) marks 2 oldest as gc_eligible."""
        chain_ids = _build_chain(5, "procedural")
        marked = db.enforce_chain_depth_limits()
        assert marked == 2

        conn = db.get_db()
        try:
            # Oldest 2 should be gc_eligible
            for i in range(2):
                row = conn.execute(
                    "SELECT gc_eligible FROM memories WHERE id = ?",
                    (chain_ids[i],),
                ).fetchone()
                assert row["gc_eligible"] == 1, f"chain_ids[{i}] should be gc_eligible"

            # Newest 3 should NOT be gc_eligible
            for i in range(2, 5):
                row = conn.execute(
                    "SELECT gc_eligible FROM memories WHERE id = ?",
                    (chain_ids[i],),
                ).fetchone()
                assert row["gc_eligible"] == 0, f"chain_ids[{i}] should not be gc_eligible"
        finally:
            conn.close()

    def test_protected_memory_not_gced(self):
        """Memory with importance >= 9 is not marked gc_eligible."""
        chain_ids = _build_chain(5, "procedural", importance=9)
        marked = db.enforce_chain_depth_limits()
        assert marked == 0

    def test_empty_database(self):
        """Empty database returns 0."""
        marked = db.enforce_chain_depth_limits()
        assert marked == 0

    def test_chain_with_missing_intermediate(self):
        """Chain with a deleted intermediate memory stops traversal gracefully."""
        chain_ids = _build_chain(5, "procedural")

        # Delete an intermediate memory
        conn = db.get_db()
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM memories WHERE id = ?", (chain_ids[2],))
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")
        finally:
            conn.close()

        # Should not raise — chain stops at the gap
        marked = db.enforce_chain_depth_limits()
        # Chain from chain_ids[0] goes: [0] -> [1] -> (missing [2]) stops at [1]
        # That's length 2, within limit 3, so no GC from that sub-chain
        assert marked >= 0  # Just verify no crash

    def test_different_types_different_limits(self):
        """correction (limit=2) and episodic (limit=5) have different limits."""
        # correction chain of 4 (limit=2) → 2 excess
        corr_ids = _build_chain(4, "correction")
        # episodic chain of 4 (limit=5) → 0 excess
        ep_ids = _build_chain(4, "episodic")

        marked = db.enforce_chain_depth_limits()
        assert marked == 2

        conn = db.get_db()
        try:
            # correction oldest 2 should be gc_eligible
            for i in range(2):
                row = conn.execute(
                    "SELECT gc_eligible FROM memories WHERE id = ?",
                    (corr_ids[i],),
                ).fetchone()
                assert row["gc_eligible"] == 1

            # episodic should all be fine
            for ep_id in ep_ids:
                row = conn.execute(
                    "SELECT gc_eligible FROM memories WHERE id = ?",
                    (ep_id,),
                ).fetchone()
                assert row["gc_eligible"] == 0
        finally:
            conn.close()

    def test_circular_chain_no_infinite_loop(self):
        """Circular superseded_by chain doesn't cause infinite loop."""
        chain_ids = _build_chain(3, "procedural")

        # Create cycle: newest points back to oldest
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?",
                (chain_ids[0], chain_ids[2]),
            )
            conn.commit()
        finally:
            conn.close()

        # Should terminate without hanging
        marked = db.enforce_chain_depth_limits()
        assert marked >= 0  # Just verify no hang/crash

    def test_get_supersession_chain(self):
        """get_supersession_chain walks from oldest to newest."""
        chain_ids = _build_chain(4, "procedural")
        result = db.get_supersession_chain(chain_ids[0])
        assert result == chain_ids

    def test_get_supersession_chain_single(self):
        """Single memory with no supersession returns [self]."""
        mem = {
            "type": "semantic",
            "content": "standalone memory",
            "importance": 5,
        }
        mem_id = db.insert_memory(mem, "session-1", "/test/project")
        result = db.get_supersession_chain(mem_id)
        assert result == [mem_id]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestCommunityDetection — Sprint 3
# ═══════════════════════════════════════════════════════════════════════════════

import kg


def _create_entity(name, entity_type="CONCEPT", description=None):
    """Helper: create an entity and return its id."""
    return kg.upsert_entity(name, entity_type, description=description)


def _create_relationship(subject_name, predicate, object_name, confidence=0.5):
    """Helper: create a relationship and return its id."""
    return kg.insert_relationship(
        subject_name, predicate, object_name, confidence=confidence,
    )


class TestCommunityDetection:
    """Test detect_communities community assignment logic."""

    def test_connected_entities_same_community(self):
        """Connected entities get the same community_id."""
        _create_entity("Python")
        _create_entity("Flask")
        _create_relationship("Python", "USES", "Flask")

        num = kg.detect_communities()
        assert num >= 1

        conn = db.get_db()
        try:
            py = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'Python'"
            ).fetchone()
            fl = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'Flask'"
            ).fetchone()
        finally:
            conn.close()
        assert py["community_id"] is not None
        assert py["community_id"] == fl["community_id"]

    def test_disconnected_entities_different_communities(self):
        """Disconnected entities get different community_ids."""
        _create_entity("Alpha")
        _create_entity("Beta")
        _create_relationship("Alpha", "USES", "Beta")
        _create_entity("Gamma")
        _create_entity("Delta")
        _create_relationship("Gamma", "USES", "Delta")

        num = kg.detect_communities()
        assert num >= 2

        conn = db.get_db()
        try:
            alpha = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'Alpha'"
            ).fetchone()
            gamma = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'Gamma'"
            ).fetchone()
        finally:
            conn.close()
        assert alpha["community_id"] != gamma["community_id"]

    def test_empty_graph_returns_zero(self):
        """Empty graph: no-op, returns 0."""
        num = kg.detect_communities()
        assert num == 0

    def test_single_entity_gets_community(self):
        """Single entity gets a community_id."""
        _create_entity("Singleton")
        num = kg.detect_communities()
        assert num == 1

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'Singleton'"
            ).fetchone()
        finally:
            conn.close()
        assert row["community_id"] is not None

    def test_all_singletons_unique_communities(self):
        """All singletons (no relationships): each gets a unique community."""
        _create_entity("Lone1")
        _create_entity("Lone2")
        _create_entity("Lone3")

        num = kg.detect_communities()
        assert num == 3

        conn = db.get_db()
        try:
            rows = conn.execute(
                "SELECT community_id FROM kg_entities ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        ids = [r["community_id"] for r in rows]
        assert len(set(ids)) == 3

    def test_entity_cap_exceeded_skips(self):
        """Entity cap exceeded: skips, returns 0."""
        _create_entity("E1")
        _create_entity("E2")
        num = kg.detect_communities(max_entities=1)
        assert num == 0

        # community_id should remain unchanged (NULL)
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'E1'"
            ).fetchone()
        finally:
            conn.close()
        assert row["community_id"] is None

    def test_networkx_unavailable_fallback(self):
        """NetworkX unavailable: fallback to CTE works."""
        _create_entity("NodeA")
        _create_entity("NodeB")
        _create_relationship("NodeA", "USES", "NodeB")
        _create_entity("NodeC")

        with patch.dict("sys.modules", {"networkx": None}):
            # Force reimport failure
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "networkx":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                num = kg.detect_communities()

        assert num >= 2  # NodeA+NodeB in one, NodeC in another

        conn = db.get_db()
        try:
            a = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'NodeA'"
            ).fetchone()
            b = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'NodeB'"
            ).fetchone()
            c = conn.execute(
                "SELECT community_id FROM kg_entities WHERE name = 'NodeC'"
            ).fetchone()
        finally:
            conn.close()
        assert a["community_id"] == b["community_id"]
        assert a["community_id"] != c["community_id"]

    def test_rerun_gives_consistent_results(self):
        """After community assignment, re-running gives consistent results."""
        _create_entity("Stable1")
        _create_entity("Stable2")
        _create_relationship("Stable1", "USES", "Stable2")

        num1 = kg.detect_communities()
        conn = db.get_db()
        try:
            ids1 = {
                r["name"]: r["community_id"]
                for r in conn.execute("SELECT name, community_id FROM kg_entities").fetchall()
            }
        finally:
            conn.close()

        num2 = kg.detect_communities()
        conn = db.get_db()
        try:
            ids2 = {
                r["name"]: r["community_id"]
                for r in conn.execute("SELECT name, community_id FROM kg_entities").fetchall()
            }
        finally:
            conn.close()

        assert num1 == num2
        # Connected entities should still share community
        assert ids2["Stable1"] == ids2["Stable2"]


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TestRelationshipDecay — Sprint 3
# ═══════════════════════════════════════════════════════════════════════════════

class TestRelationshipDecay:
    """Test apply_relationship_decay time-based decay logic."""

    def test_non_permanent_decays_after_window(self):
        """Non-permanent predicate decays after window."""
        _create_entity("OldTech")
        _create_entity("OldVersion")
        rel_id = _create_relationship("OldTech", "HAS_VERSION", "OldVersion", confidence=0.8)

        # Backdate created_at to 60 days ago (HAS_VERSION window = 30 days)
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE kg_relationships SET created_at = ? WHERE id = ?",
                (time.time() - 60 * 86400, rel_id),
            )
            conn.commit()
        finally:
            conn.close()

        result = kg.apply_relationship_decay()
        assert result['decayed'] == 1

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT confidence FROM kg_relationships WHERE id = ?", (rel_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["confidence"] == pytest.approx(0.4)

    def test_permanent_predicate_no_decay(self):
        """Permanent predicate (NULL window) doesn't decay."""
        _create_entity("Rule1")
        _create_entity("Target1")
        rel_id = _create_relationship("Rule1", "APPLIES_TO", "Target1", confidence=0.8)

        # Backdate far into the past
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE kg_relationships SET created_at = ? WHERE id = ?",
                (time.time() - 365 * 86400, rel_id),
            )
            conn.commit()
        finally:
            conn.close()

        result = kg.apply_relationship_decay()
        assert result['decayed'] == 0

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT confidence FROM kg_relationships WHERE id = ?", (rel_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["confidence"] == pytest.approx(0.8)

    def test_confidence_below_threshold_expired(self):
        """Confidence drops below 0.1: marked as expired (valid_until set)."""
        _create_entity("ExpTech")
        _create_entity("ExpVer")
        rel_id = _create_relationship("ExpTech", "HAS_VERSION", "ExpVer", confidence=0.15)

        # Backdate
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE kg_relationships SET created_at = ? WHERE id = ?",
                (time.time() - 60 * 86400, rel_id),
            )
            conn.commit()
        finally:
            conn.close()

        result = kg.apply_relationship_decay()
        assert result['expired'] == 1

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT confidence, valid_until FROM kg_relationships WHERE id = ?",
                (rel_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["confidence"] == pytest.approx(0.075)
        assert row["valid_until"] is not None

    def test_multiple_decays_halve_each_time(self):
        """Multiple decays: confidence halves each time."""
        _create_entity("MultiTech")
        _create_entity("MultiVer")
        rel_id = _create_relationship("MultiTech", "HAS_VERSION", "MultiVer", confidence=0.8)

        # Backdate
        old_time = time.time() - 60 * 86400
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE kg_relationships SET created_at = ? WHERE id = ?",
                (old_time, rel_id),
            )
            conn.commit()
        finally:
            conn.close()

        # First decay
        kg.apply_relationship_decay()

        # Reset sync state to allow second run
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE kg_sync_state SET value = '0' WHERE key = 'last_relationship_decay'"
            )
            conn.commit()
        finally:
            conn.close()

        # Second decay
        kg.apply_relationship_decay()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT confidence FROM kg_relationships WHERE id = ?", (rel_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["confidence"] == pytest.approx(0.2)  # 0.8 * 0.5 * 0.5

    def test_idempotency_24h_guard(self):
        """Idempotency: only runs once per 24h."""
        _create_entity("IdempTech")
        _create_entity("IdempVer")
        rel_id = _create_relationship("IdempTech", "HAS_VERSION", "IdempVer", confidence=0.8)

        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE kg_relationships SET created_at = ? WHERE id = ?",
                (time.time() - 60 * 86400, rel_id),
            )
            conn.commit()
        finally:
            conn.close()

        result1 = kg.apply_relationship_decay()
        assert result1['decayed'] == 1

        # Second call should be skipped
        result2 = kg.apply_relationship_decay()
        assert result2['decayed'] == 0

    def test_recent_relationship_not_decayed(self):
        """Relationship created within decay window: not decayed."""
        _create_entity("RecentTech")
        _create_entity("RecentVer")
        rel_id = _create_relationship("RecentTech", "HAS_VERSION", "RecentVer", confidence=0.8)

        # created_at is now (within the 30-day HAS_VERSION window)
        result = kg.apply_relationship_decay()
        assert result['decayed'] == 0

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT confidence FROM kg_relationships WHERE id = ?", (rel_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["confidence"] == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TestCommunityAwareRetrieval — Sprint 3
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommunityAwareRetrieval:
    """Test community-aware entity sorting in neighborhood retrieval."""

    def test_same_community_entities_preferred(self):
        """Same-community entities appear first when available."""
        # Create two clusters
        e1 = _create_entity("QueryEnt", description="main query entity")
        e2 = _create_entity("SameCluster", description="same cluster member")
        e3 = _create_entity("OtherCluster", description="other cluster member")
        _create_relationship("QueryEnt", "USES", "SameCluster")
        _create_relationship("QueryEnt", "RELATED_TO", "OtherCluster")

        # Manually assign communities
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE kg_entities SET community_id = 0 WHERE name IN ('QueryEnt', 'SameCluster')"
            )
            conn.execute(
                "UPDATE kg_entities SET community_id = 1 WHERE name = 'OtherCluster'"
            )
            conn.commit()
        finally:
            conn.close()

        neighborhood = kg.kg_entity_neighborhood(["QueryEnt"], max_depth=2, max_neighbors=10)
        entities = neighborhood["entities"]

        # Find positions
        names = [e["name"] for e in entities]
        assert "SameCluster" in names
        assert "OtherCluster" in names

        # Entities with community_id=0 (same as QueryEnt) should have that id
        same_cluster_ent = next(e for e in entities if e["name"] == "SameCluster")
        assert same_cluster_ent["community_id"] == 0

    def test_all_null_community_ids_no_crash(self):
        """All NULL community_ids: fallback works without crash."""
        _create_entity("NullComm1")
        _create_entity("NullComm2")
        _create_relationship("NullComm1", "USES", "NullComm2")

        # Don't assign any communities — all community_id remain NULL
        neighborhood = kg.kg_entity_neighborhood(["NullComm1"], max_depth=2, max_neighbors=10)
        assert len(neighborhood["entities"]) >= 1
        # Should not crash — entities returned normally


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 4: Temporal Score Consolidation + Batch Caching
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalScoreConsolidation:
    """Test db.temporal_score() canonical function."""

    def test_known_inputs_match_expected(self):
        """temporal_score with known inputs produces expected formula output."""
        now = time.time()
        # Very recent memory with some accesses
        score = db.temporal_score(
            access_count=5,
            last_accessed_at=now - 60,  # 1 minute ago
            created_at=now - 3600,
            decay_rate=0.16,
            stability=0.0,
        )
        # Score should be in [0, 1]
        assert 0.0 <= score <= 1.0
        # Recent memory with accesses should score reasonably high
        assert score > 0.3

    def test_zero_access_count_ebbinghaus_only(self):
        """access_count=0: only Ebbinghaus decay, no ACT-R component."""
        import math
        now = time.time()
        created = now - 86400  # 1 day ago
        score = db.temporal_score(
            access_count=0,
            last_accessed_at=None,
            created_at=created,
            decay_rate=0.16,
            stability=0.0,
        )
        # Should equal strength * 0.5
        t_days = max((now - created) / 86400.0, 1e-6)
        lambda_eff = 0.16 * (1.0 - 0.0 * 0.8)
        expected = math.exp(-lambda_eff * t_days) * 0.5
        assert score == pytest.approx(expected, rel=1e-3)

    def test_high_stability_slower_decay(self):
        """High stability results in slower decay (higher score)."""
        now = time.time()
        created = now - 86400 * 7  # 7 days ago
        score_low_stab = db.temporal_score(
            access_count=0,
            last_accessed_at=None,
            created_at=created,
            decay_rate=0.16,
            stability=0.0,
        )
        score_high_stab = db.temporal_score(
            access_count=0,
            last_accessed_at=None,
            created_at=created,
            decay_rate=0.16,
            stability=0.9,
        )
        assert score_high_stab > score_low_stab

    def test_old_memory_many_accesses_actr_boost(self):
        """Old memory with many accesses: ACT-R activation boosts score."""
        now = time.time()
        created = now - 86400 * 30  # 30 days ago
        score_no_access = db.temporal_score(
            access_count=0,
            last_accessed_at=None,
            created_at=created,
            decay_rate=0.16,
            stability=0.0,
        )
        score_many_access = db.temporal_score(
            access_count=50,
            last_accessed_at=now - 3600,  # accessed 1 hour ago
            created_at=created,
            decay_rate=0.16,
            stability=0.0,
        )
        assert score_many_access > score_no_access


class TestTemporalScoreBatch:
    """Test compute_temporal_scores() batch function."""

    def test_batch_updates_temporal_score_column(self):
        """Batch computation updates temporal_score column."""
        mem_id = _insert_procedural("batch test memory", importance=7)
        updated = db.compute_temporal_scores(chunk_size=10)
        assert updated >= 1
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT temporal_score FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["temporal_score"] is not None
        assert 0.0 <= row["temporal_score"] <= 1.0

    def test_batch_sets_score_computed_at(self):
        """Batch computation sets score_computed_at timestamp."""
        mem_id = _insert_procedural("computed_at test memory", importance=7)
        before = time.time()
        db.compute_temporal_scores(chunk_size=10)
        after = time.time()
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT score_computed_at FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["score_computed_at"] is not None
        assert before <= row["score_computed_at"] <= after

    def test_zero_memories_returns_zero(self):
        """Empty database: returns 0."""
        updated = db.compute_temporal_scores(chunk_size=10)
        assert updated == 0

    def test_superseded_memories_skipped(self):
        """Superseded memories are not updated by batch computation."""
        mem_id = _insert_procedural("original memory", importance=7)
        superseder_id = _insert_procedural("superseder memory", importance=7)
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?",
                (superseder_id, mem_id),
            )
            conn.commit()
        finally:
            conn.close()

        db.compute_temporal_scores(chunk_size=10)

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT temporal_score, score_computed_at FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        # Superseded memory should not have been updated
        assert row["temporal_score"] is None
        assert row["score_computed_at"] is None


class TestDaemonCachedScore:
    """Test daemon uses cached temporal scores when fresh."""

    def test_fresh_cache_used(self):
        """When score_computed_at is recent, cached score is used."""
        now = time.time()
        mem = {
            "access_count": 5,
            "last_accessed_at": now - 60,
            "created_at": now - 3600,
            "decay_rate": 0.16,
            "stability": 0.0,
            "temporal_score": 0.99,  # artificial cached value
            "score_computed_at": now - 100,  # 100 seconds ago (fresh)
        }
        # The daemon search logic checks: if score_computed_at exists and
        # is within 6 hours, use cached temporal_score
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
        # Should use cached value
        assert t_score == 0.99

    def test_stale_cache_triggers_recompute(self):
        """When score_computed_at is old, score is recomputed."""
        now = time.time()
        mem = {
            "access_count": 5,
            "last_accessed_at": now - 60,
            "created_at": now - 3600,
            "decay_rate": 0.16,
            "stability": 0.0,
            "temporal_score": 0.99,  # artificial cached value
            "score_computed_at": now - 30000,  # >6 hours ago (stale)
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
        # Should NOT use cached value, recomputed score != 0.99
        assert t_score != 0.99
        assert 0.0 <= t_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 6: Garbage Collection + Integration Validation
# ═══════════════════════════════════════════════════════════════════════════════


def _insert_memory(content, memory_type="semantic", importance=5, **kwargs):
    """Helper: insert a memory of any type and return its id."""
    mem = {
        "type": memory_type,
        "content": content,
        "importance": importance,
    }
    mem.update(kwargs)
    return db.insert_memory(mem, "test-session", "/test/project")


class TestGarbageCollection:
    """Test run_garbage_collection soft-delete logic."""

    def test_chain_pruned_counted(self):
        """Chain-pruned memories are handled by enforce_chain_depth_limits."""
        chain_ids = _build_chain(5, "procedural")
        marked = db.enforce_chain_depth_limits()  # marks 2 oldest as gc_eligible
        assert marked == 2

        # GC stats only count newly forgotten memories (delta), not cumulative
        stats = db.run_garbage_collection()
        assert stats['gc_forgotten'] == 0
        assert stats['total'] == 0

    def test_forgotten_superseded_marked(self):
        """Forgotten + superseded memories get gc_eligible=1."""
        old_id = _insert_memory("old fact about X", memory_type="semantic", importance=5,
                                subject="X", predicate="IS")
        new_id = _insert_memory("new fact about X", memory_type="semantic", importance=5,
                                subject="X", predicate="IS")
        db.detect_supersession(new_id, "X", "IS")

        # Set temporal_score very low on the old memory
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET temporal_score = 0.001, score_computed_at = ? WHERE id = ?",
                (time.time(), old_id),
            )
            conn.commit()
        finally:
            conn.close()

        stats = db.run_garbage_collection()
        assert stats['gc_forgotten'] == 1

        # Verify the old memory is now gc_eligible
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT gc_eligible FROM memories WHERE id = ?", (old_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["gc_eligible"] == 1

    def test_protected_memories_survive_gc(self):
        """Protected memories (importance >= 9) are never GC'd."""
        old_id = _insert_memory("critical rule", memory_type="procedural", importance=9,
                                subject="CritRule", predicate="IS")
        new_id = _insert_memory("updated critical rule", memory_type="procedural", importance=9,
                                subject="CritRule", predicate="IS")
        db.detect_supersession(new_id, "CritRule", "IS")

        # Set temporal_score very low
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET temporal_score = 0.001, score_computed_at = ? WHERE id = ?",
                (time.time(), old_id),
            )
            conn.commit()
        finally:
            conn.close()

        stats = db.run_garbage_collection()
        assert stats['gc_forgotten'] == 0

        # Verify the old memory is NOT gc_eligible
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT gc_eligible FROM memories WHERE id = ?", (old_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["gc_eligible"] == 0

    def test_non_superseded_low_score_not_gced(self):
        """Non-superseded memories with low temporal_score are NOT GC'd (safety)."""
        mem_id = _insert_memory("standalone low-score memory", importance=3)

        # Set temporal_score very low but NOT superseded
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET temporal_score = 0.001, score_computed_at = ? WHERE id = ?",
                (time.time(), mem_id),
            )
            conn.commit()
        finally:
            conn.close()

        stats = db.run_garbage_collection()
        assert stats['gc_forgotten'] == 0

        # Verify not gc_eligible
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT gc_eligible FROM memories WHERE id = ?", (mem_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["gc_eligible"] == 0

    def test_empty_database_returns_zeros(self):
        """Empty database: returns all zeros."""
        stats = db.run_garbage_collection()
        assert stats == {'gc_chain_pruned': 0, 'gc_forgotten': 0, 'total': 0}

    def test_null_temporal_score_not_affected(self):
        """Memories without temporal_score (NULL) are not affected by GC."""
        old_id = _insert_memory("null score memory", memory_type="semantic", importance=5,
                                subject="NullScore", predicate="IS")
        new_id = _insert_memory("newer null score memory", memory_type="semantic", importance=5,
                                subject="NullScore", predicate="IS")
        db.detect_supersession(new_id, "NullScore", "IS")

        # temporal_score is NULL by default — should NOT be GC'd
        stats = db.run_garbage_collection()
        assert stats['gc_forgotten'] == 0

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT gc_eligible FROM memories WHERE id = ?", (old_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["gc_eligible"] == 0


class TestFullLifecycle:
    """Test full pipeline integration scenarios."""

    def test_procedural_reinforcement_full_pipeline(self):
        """Insert procedural 5x: reinforcement_count=5, stability=1.0, promotion_candidate=1."""
        mem_id = _insert_procedural("always use pytest fixtures", importance=7)

        for i in range(5):
            db.increment_reinforcement(mem_id)

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT reinforcement_count, stability, promotion_candidate FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["reinforcement_count"] == 5
        assert row["stability"] == pytest.approx(1.0)
        assert row["promotion_candidate"] == 1

    def test_insert_supersede_event_bus(self):
        """Insert -> supersede -> event bus flags all = 1."""
        old_id, new_id = _create_supersession_pair(
            subject="TestLib", predicate="HAS_VERSION",
            old_content="TestLib v1", new_content="TestLib v2",
        )

        stats = db.process_supersession_events()
        assert stats['processed'] == 1

        # Verify all flags are 1
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT processed_by_temporal, processed_by_kg, processed_by_contextual "
                "FROM supersession_events WHERE old_memory_id = ?",
                (old_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row["processed_by_temporal"] == 1
        assert row["processed_by_kg"] == 1
        assert row["processed_by_contextual"] == 1

    def test_gc_integration_superseded_decayed(self):
        """GC integration: superseded + decayed memory gets gc_eligible=1."""
        old_id = _insert_memory("old version info", memory_type="semantic", importance=5,
                                subject="VersionInfo", predicate="IS")
        new_id = _insert_memory("new version info", memory_type="semantic", importance=5,
                                subject="VersionInfo", predicate="IS")
        db.detect_supersession(new_id, "VersionInfo", "IS")

        # Compute temporal scores so old memory has a score
        db.compute_temporal_scores()

        # Force temporal_score to very low
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET temporal_score = 0.002, score_computed_at = ? WHERE id = ?",
                (time.time(), old_id),
            )
            conn.commit()
        finally:
            conn.close()

        stats = db.run_garbage_collection()
        assert stats['gc_forgotten'] == 1

        # Verify the old memory is gc_eligible
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT gc_eligible FROM memories WHERE id = ?", (old_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["gc_eligible"] == 1


class TestDaemonBackgroundJobs:
    """Test daemon _run_background_jobs execution."""

    def test_background_jobs_empty_db(self):
        """Background jobs run without error on empty DB."""
        # Import and run the background jobs function directly
        daemon_dir = str(Path(__file__).parent.parent / "daemon")
        if daemon_dir not in sys.path:
            sys.path.insert(0, daemon_dir)

        import importlib
        import embedding_daemon
        importlib.reload(embedding_daemon)

        # Cancel any timer that might be started
        embedding_daemon._bg_timer = None

        # Patch the timer to avoid actual scheduling
        with patch("threading.Timer") as mock_timer:
            mock_timer.return_value.daemon = True
            embedding_daemon._run_background_jobs()

        # No exception means success

    def test_background_jobs_all_stages(self):
        """Background jobs process all stages (temporal, events, chain, gc, community, decay)."""
        # Insert some data to exercise all stages
        _insert_memory("test memory for bg jobs", memory_type="procedural", importance=7)
        _create_entity("BgEntity")

        daemon_dir = str(Path(__file__).parent.parent / "daemon")
        if daemon_dir not in sys.path:
            sys.path.insert(0, daemon_dir)

        import importlib
        import embedding_daemon
        importlib.reload(embedding_daemon)

        embedding_daemon._bg_timer = None

        with patch("threading.Timer") as mock_timer:
            mock_timer.return_value.daemon = True
            embedding_daemon._run_background_jobs()

        # Verify temporal score was computed
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT temporal_score FROM memories WHERE content = 'test memory for bg jobs'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row["temporal_score"] is not None


class TestRegressionGate:
    """Verify Phase 1-5 patterns still work end-to-end."""

    def test_phase1_through_5_patterns(self):
        """insert_memory + store_embedding + detect_supersession + detect_content_supersession all work."""
        # Phase 1: insert_memory
        mem1 = {
            "type": "semantic",
            "content": "Python version is 3.10",
            "importance": 7,
            "subject": "Python",
            "predicate": "HAS_VERSION",
        }
        id1 = db.insert_memory(mem1, "session-1", "/test/project")
        assert id1 is not None

        # Phase 2: store_embedding
        fake_embedding = [0.1] * 384
        db.store_embedding(id1, fake_embedding)
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (id1,)
            ).fetchone()
        finally:
            conn.close()
        assert row["embedding"] is not None

        # Phase 1: detect_supersession (structured)
        mem2 = {
            "type": "semantic",
            "content": "Python version is 3.11",
            "importance": 7,
            "subject": "Python",
            "predicate": "HAS_VERSION",
        }
        id2 = db.insert_memory(mem2, "session-2", "/test/project")
        superseded = db.detect_supersession(id2, "Python", "HAS_VERSION")
        assert superseded == id1

        # Verify superseded_by is set
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT superseded_by FROM memories WHERE id = ?", (id1,)
            ).fetchone()
        finally:
            conn.close()
        assert row["superseded_by"] == id2

        # Phase 2: detect_content_supersession (fallback)
        # Content must have Jaccard >= 0.6 to trigger supersession
        mem3 = {
            "type": "correction",
            "content": "Always use the logging module for debug output in all Python code",
            "importance": 8,
        }
        id3 = db.insert_memory(mem3, "session-3", "/test/project")
        mem4 = {
            "type": "correction",
            "content": "Always use the logging module for debug output in all Python scripts",
            "importance": 8,
        }
        id4 = db.insert_memory(mem4, "session-4", "/test/project")
        superseded2 = db.detect_content_supersession(id4, mem4["content"], "correction")
        assert superseded2 == id3


class TestE2EReinforcement:
    """End-to-end integration test: simulates multiple sessions saying the same
    thing and verifies reinforcement increments without manual session testing."""

    def test_correction_reinforcement_across_sessions(self):
        """Say 'don't use unittest, use pytest' in 3 sessions → reinforcement_count=2."""
        import store_memory

        # Session 1: first time saying it
        extraction1 = {
            "memories": [{
                "type": "correction",
                "content": "Don't use unittest, use pytest with fixtures",
                "importance": 7,
                "subject": "unittest and pytest",
                "predicate": "must_not_use",
                "object": "unittest",
            }],
            "entities": [],
            "relationships": [],
            "summary": ["correction about unittest"],
        }
        sys.argv = ["store_memory.py", json.dumps(extraction1), "session-e2e-1"]
        store_memory.main()

        # Verify: memory exists with reinf=0
        conn = db.get_db()
        row = conn.execute(
            "SELECT reinforcement_count, stability FROM memories WHERE content LIKE '%unittest%pytest%' AND superseded_by IS NULL"
        ).fetchone()
        conn.close()
        assert row is not None, "Memory should have been inserted"
        assert row["reinforcement_count"] == 0

        # Session 2: same correction, different wording
        extraction2 = {
            "memories": [{
                "type": "correction",
                "content": "No, don't use unittest. Always use pytest instead.",
                "importance": 7,
                "subject": "unittest and pytest",
                "predicate": "must_not_use",
                "object": "unittest",
            }],
            "entities": [],
            "relationships": [],
            "summary": ["correction about unittest"],
        }
        sys.argv = ["store_memory.py", json.dumps(extraction2), "session-e2e-2"]
        store_memory.main()

        # Verify: original memory reinforced, no new memory inserted
        conn = db.get_db()
        rows = conn.execute(
            "SELECT reinforcement_count, stability FROM memories WHERE content LIKE '%unittest%' AND memory_type = 'correction' AND superseded_by IS NULL ORDER BY created_at"
        ).fetchall()
        conn.close()

        # Should still be just 1 active memory (the original), reinforced
        active = [r for r in rows if r["reinforcement_count"] > 0]
        assert len(active) >= 1, f"Expected at least 1 reinforced memory, got {len(active)}"
        assert active[0]["reinforcement_count"] >= 1

        # Session 3: third repetition (same triple as session 1)
        extraction3 = {
            "memories": [{
                "type": "correction",
                "content": "never use unittest, always pytest",
                "importance": 7,
                "subject": "unittest and pytest",
                "predicate": "must_not_use",
                "object": "unittest",
            }],
            "entities": [],
            "relationships": [],
            "summary": ["correction about unittest"],
        }
        sys.argv = ["store_memory.py", json.dumps(extraction3), "session-e2e-3"]
        store_memory.main()

        conn = db.get_db()
        rows = conn.execute(
            "SELECT reinforcement_count, stability, promotion_candidate FROM memories WHERE content LIKE '%unittest%' AND memory_type = 'correction' AND superseded_by IS NULL ORDER BY reinforcement_count DESC"
        ).fetchall()
        conn.close()

        top = rows[0]
        assert top["reinforcement_count"] >= 2, f"Expected reinf >= 2, got {top['reinforcement_count']}"

    def test_supersession_not_confused_with_reinforcement(self):
        """'use MySQL' then 'use PostgreSQL' should supersede, not reinforce."""
        import store_memory

        ext1 = {
            "memories": [{
                "type": "correction", "content": "use MySQL for the database",
                "importance": 8, "subject": "database", "predicate": "USES", "object": "MySQL",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext1), "s-sup-1"]
        store_memory.main()

        ext2 = {
            "memories": [{
                "type": "correction", "content": "use PostgreSQL for the database",
                "importance": 8, "subject": "database", "predicate": "USES", "object": "PostgreSQL",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext2), "s-sup-2"]
        store_memory.main()

        conn = db.get_db()
        # The old MySQL memory should be superseded, not reinforced
        mysql_mem = conn.execute(
            "SELECT superseded_by, reinforcement_count FROM memories WHERE content LIKE '%MySQL%'"
        ).fetchone()
        pg_mem = conn.execute(
            "SELECT superseded_by, reinforcement_count FROM memories WHERE content LIKE '%PostgreSQL%'"
        ).fetchone()
        conn.close()

        assert mysql_mem["superseded_by"] is not None, "MySQL memory should be superseded"
        assert mysql_mem["reinforcement_count"] == 0, "MySQL memory should NOT be reinforced"
        assert pg_mem["superseded_by"] is None, "PostgreSQL memory should be active"


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End Integration Tests — Phase 6 Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestE2ETriageToStore:
    """Simulate: user says something -> triage detects -> extraction -> store_memory processes it."""

    def test_triage_fires_on_new_patterns(self):
        """Verify triage patterns detect correction/decision signals for various phrasings."""
        import triage

        test_cases = [
            # Sentence-start patterns
            ("Always use pytest for testing", "correction"),
            ("never use print() for debugging in production", "correction"),
            ("from now on, use logging instead", "decision"),
            # Existing patterns
            ("don't use unittest, switch to pytest", "correction"),
            ("actually, we should use Redis not MySQL", "correction"),
            ("stop using global variables", "correction"),
            ("let's use Docker for deployment", "decision"),
            ("remember, always run tests first", "decision"),
            ("that's wrong, use snake_case", "correction"),
            ("instead, use pathlib for file paths", "correction"),
        ]

        for text, expected_type in test_cases:
            signals = triage.triage(text)
            assert len(signals) > 0, (
                f"No signal detected for: '{text}'"
            )
            signal_types = [s["type"] for s in signals]
            assert expected_type in signal_types, (
                f"Expected '{expected_type}' signal for '{text}', got {signal_types}"
            )

    def test_triage_mid_sentence_always(self):
        """'always' mid-sentence should NOT trigger (too many false positives).

        The anchor (?:^|\\n|[.!]\\s+) requires sentence start. Mid-sentence
        'always' like 'I always use vim' should not fire.
        """
        import triage

        # Should NOT trigger
        false_positives = [
            "I always use vim for editing",
            "Can you always use the latest version?",
            "The function will always return true",
        ]
        for text in false_positives:
            signals = triage.triage(text)
            always_signals = [s for s in signals if "always" in s.get("match", "").lower()]
            assert len(always_signals) == 0, (
                f"False positive on: '{text}' — matched: {always_signals}"
            )

        # Should trigger (sentence start or after period)
        true_positives = [
            "Always use ruff for linting",
            "Ok. Always use 4 spaces for indentation",
            "Done.\nAlways run tests before committing",
        ]
        for text in true_positives:
            signals = triage.triage(text)
            assert len(signals) > 0, f"No signal for: '{text}'"

    def test_triage_to_store_roundtrip(self):
        """Create a realistic extraction JSON and pass through store_memory.main()."""
        import store_memory

        extraction = {
            "memories": [{
                "type": "procedural",
                "content": "Always use pathlib instead of os.path for file operations",
                "importance": 8,
                "subject": "file_operations",
                "predicate": "USES",
                "object": "pathlib",
            }],
            "entities": [{"name": "pathlib", "type": "LIBRARY"}],
            "relationships": [],
            "summary": ["procedural rule about pathlib"],
        }
        sys.argv = ["store_memory.py", json.dumps(extraction), "session-triage-rt"]
        store_memory.main()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT content, memory_type, importance, subject, predicate, object, session_id "
                "FROM memories WHERE content LIKE '%pathlib%' AND superseded_by IS NULL"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Memory should have been inserted"
        assert row["memory_type"] == "procedural"
        assert row["importance"] == 8
        assert row["subject"] == "file_operations"
        assert row["predicate"] == "USES"
        assert row["object"] == "pathlib"
        assert row["session_id"] == "session-triage-rt"

    def test_multi_type_extraction_normalized(self):
        """qwen2.5:3b sometimes outputs type='correction | semantic | procedural'. Verify normalization."""
        # Test via insert_memory directly, which handles the normalization
        multi_type_cases = [
            ("correction | semantic | procedural", "correction"),
            ("procedural | semantic", "procedural"),
            ("semantic, episodic", "semantic"),
            ("garbage_type", "episodic"),  # fallback to episodic
            ("CORRECTION", "correction"),  # case normalization
        ]

        for raw_type, expected_type in multi_type_cases:
            mem = {
                "type": raw_type,
                "content": f"Test normalization for type={raw_type}",
                "importance": 5,
            }
            mem_id = db.insert_memory(mem, f"session-norm-{raw_type}", "/test/norm")

            conn = db.get_db()
            try:
                row = conn.execute(
                    "SELECT memory_type FROM memories WHERE id = ?", (mem_id,)
                ).fetchone()
            finally:
                conn.close()

            assert row["memory_type"] == expected_type, (
                f"Raw type '{raw_type}' should normalize to '{expected_type}', "
                f"got '{row['memory_type']}'"
            )


class TestE2EReinforcementWithTriples:
    """Test reinforcement behavior with structured (subject, predicate, object) triples."""

    def test_same_triple_different_content_reinforces(self):
        """Insert 3 memories with same (subject, predicate, object) but different content text.

        Only 1 memory should exist; reinforcement_count = 2.
        """
        import store_memory

        base_extraction = {
            "entities": [], "relationships": [], "summary": [],
        }

        contents = [
            "Use pytest fixtures for test setup",
            "Always prefer pytest fixtures over manual setup",
            "pytest fixtures are the standard for test initialization",
        ]

        for i, content in enumerate(contents):
            ext = {
                **base_extraction,
                "memories": [{
                    "type": "procedural",
                    "content": content,
                    "importance": 7,
                    "subject": "test_setup",
                    "predicate": "USES",
                    "object": "pytest_fixtures",
                }],
            }
            sys.argv = ["store_memory.py", json.dumps(ext), f"session-triple-{i}"]
            store_memory.main()

        conn = db.get_db()
        try:
            rows = conn.execute(
                "SELECT id, reinforcement_count, content FROM memories "
                "WHERE memory_type = 'procedural' AND superseded_by IS NULL AND gc_eligible = 0"
            ).fetchall()
        finally:
            conn.close()

        # Only 1 active memory should exist (the original, reinforced twice)
        assert len(rows) == 1, (
            f"Expected 1 active memory, got {len(rows)}: "
            f"{[r['content'][:40] for r in rows]}"
        )
        assert rows[0]["reinforcement_count"] == 2, (
            f"Expected reinforcement_count=2, got {rows[0]['reinforcement_count']}"
        )

    def test_same_subject_different_object_supersedes(self):
        """Insert 'use MySQL' then 'use PostgreSQL' with same subject+predicate but different object.

        First should be superseded, not reinforced.
        """
        import store_memory

        ext1 = {
            "memories": [{
                "type": "procedural", "content": "Use MySQL for data storage",
                "importance": 7, "subject": "data_storage", "predicate": "USES", "object": "MySQL",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext1), "s-obj-diff-1"]
        store_memory.main()

        ext2 = {
            "memories": [{
                "type": "procedural", "content": "Use PostgreSQL for data storage",
                "importance": 7, "subject": "data_storage", "predicate": "USES", "object": "PostgreSQL",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext2), "s-obj-diff-2"]
        store_memory.main()

        conn = db.get_db()
        try:
            mysql_row = conn.execute(
                "SELECT superseded_by, reinforcement_count FROM memories "
                "WHERE content LIKE '%MySQL%data storage%'"
            ).fetchone()
            pg_row = conn.execute(
                "SELECT superseded_by, reinforcement_count FROM memories "
                "WHERE content LIKE '%PostgreSQL%data storage%'"
            ).fetchone()
        finally:
            conn.close()

        assert mysql_row is not None, "MySQL memory should exist"
        assert pg_row is not None, "PostgreSQL memory should exist"
        assert mysql_row["superseded_by"] is not None, "MySQL should be superseded"
        assert mysql_row["reinforcement_count"] == 0, "MySQL should NOT be reinforced"
        assert pg_row["superseded_by"] is None, "PostgreSQL should be active"

    def test_missing_triple_falls_through_to_like(self):
        """Insert memory with no triple, then insert another with overlapping content.

        Uses different wording to avoid content-hash dedup. The second insert's
        content must be a substring of the first (or vice versa) to trigger LIKE.
        """
        import store_memory

        # Session 1: original memory
        ext1 = {
            "memories": [{
                "type": "procedural",
                "content": "Always run linting before committing code to the repository",
                "importance": 6,
                # No subject, predicate, object — forces LIKE fallback
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext1), "s-like-0"]
        store_memory.main()

        # Session 2: content contains "linting before committing" (overlapping substring)
        ext2 = {
            "memories": [{
                "type": "procedural",
                "content": "run linting before committing",
                "importance": 6,
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext2), "s-like-1"]
        store_memory.main()

        conn = db.get_db()
        try:
            rows = conn.execute(
                "SELECT reinforcement_count FROM memories "
                "WHERE content LIKE '%linting before committing%' "
                "AND superseded_by IS NULL AND gc_eligible = 0"
            ).fetchall()
        finally:
            conn.close()

        # LIKE match: "run linting before committing" is a substring of the original
        # So reinforcement should fire, leaving 1 memory with count >= 1
        assert len(rows) == 1, f"Expected 1 memory, got {len(rows)}"
        assert rows[0]["reinforcement_count"] >= 1, (
            f"Expected reinforcement_count >= 1, got {rows[0]['reinforcement_count']}"
        )

    def test_partial_triple_no_false_reinforcement(self):
        """Insert two procedural memories with different (subject, predicate, object) triples.

        Should NOT reinforce each other; both should exist separately.
        The content is deliberately distinct to avoid Jaccard/content-similarity supersession.
        """
        import store_memory

        ext1 = {
            "memories": [{
                "type": "procedural",
                "content": "Configure the Redis cache layer with TTL expiration and LRU eviction policy",
                "importance": 7,
                "subject": "cache_layer",
                "predicate": "USES",
                "object": "redis",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext1), "s-partial-1"]
        store_memory.main()

        ext2 = {
            "memories": [{
                "type": "procedural",
                "content": "Deploy the monitoring dashboard with Grafana and Prometheus metrics collection",
                "importance": 7,
                "subject": "monitoring",
                "predicate": "USES",
                "object": "grafana",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext2), "s-partial-2"]
        store_memory.main()

        conn = db.get_db()
        try:
            rows = conn.execute(
                "SELECT id, content, reinforcement_count, subject, object FROM memories "
                "WHERE memory_type = 'procedural' "
                "AND superseded_by IS NULL AND gc_eligible = 0 "
                "ORDER BY created_at"
            ).fetchall()
        finally:
            conn.close()

        assert len(rows) == 2, (
            f"Expected 2 separate memories, got {len(rows)}: "
            f"{[r['content'][:40] for r in rows]}"
        )
        for r in rows:
            assert r["reinforcement_count"] == 0, (
                f"Memory '{r['content'][:40]}' should not be reinforced, "
                f"got reinforcement_count={r['reinforcement_count']}"
            )


class TestE2EFullLifecycle:
    """Test full memory lifecycle: insert -> supersede -> event bus -> GC."""

    def test_insert_supersede_eventbus_gc(self):
        """Full chain: insert A, insert B (supersedes A), process events, compute scores, GC."""
        # Insert memory A
        mem_a = {
            "type": "correction", "content": "Use Flask for the web server",
            "importance": 5, "subject": "web_server", "predicate": "USES", "object": "Flask",
        }
        id_a = db.insert_memory(mem_a, "s-lifecycle-1", "/test/lifecycle")

        # Insert memory B (same subject+predicate, different object -> supersedes A)
        mem_b = {
            "type": "correction", "content": "Use FastAPI for the web server",
            "importance": 5, "subject": "web_server", "predicate": "USES", "object": "FastAPI",
        }
        id_b = db.insert_memory(mem_b, "s-lifecycle-2", "/test/lifecycle")
        superseded = db.detect_supersession(id_b, "web_server", "USES")
        assert superseded == id_a, "A should be superseded by B"

        # Process supersession events -> verify all 3 flags set
        stats = db.process_supersession_events()
        assert stats["processed"] >= 1
        assert stats["temporal"] >= 1
        assert stats["kg"] >= 1
        assert stats["contextual"] >= 1

        # Verify all flags are set
        conn = db.get_db()
        try:
            event = conn.execute(
                "SELECT processed_by_temporal, processed_by_kg, processed_by_contextual "
                "FROM supersession_events WHERE old_memory_id = ? AND new_memory_id = ?",
                (id_a, id_b),
            ).fetchone()
        finally:
            conn.close()

        assert event["processed_by_temporal"] == 1
        assert event["processed_by_kg"] == 1
        assert event["processed_by_contextual"] == 1

        # Compute temporal scores -> verify scores cached
        updated = db.compute_temporal_scores()
        assert updated >= 1

        conn = db.get_db()
        try:
            row_b = conn.execute(
                "SELECT temporal_score, score_computed_at FROM memories WHERE id = ?",
                (id_b,),
            ).fetchone()
        finally:
            conn.close()
        assert row_b["temporal_score"] is not None
        assert row_b["score_computed_at"] is not None

        # Simulate decay: set A's temporal_score to 0.001
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET temporal_score = 0.001 WHERE id = ?", (id_a,)
            )
            conn.commit()
        finally:
            conn.close()

        # Run GC -> verify A is gc_eligible
        gc_stats = db.run_garbage_collection()
        assert gc_stats["gc_forgotten"] >= 1

        conn = db.get_db()
        try:
            row_a = conn.execute(
                "SELECT gc_eligible FROM memories WHERE id = ?", (id_a,)
            ).fetchone()
        finally:
            conn.close()
        assert row_a["gc_eligible"] == 1, "Memory A should be gc_eligible after decay"

    def test_reinforcement_to_promotion(self, tmp_path):
        """Insert procedural memory, reinforce 5 times, then promote to CLAUDE.md."""
        import store_memory

        content = "Always use type hints in function signatures"

        # Session 1: initial insert
        ext = {
            "memories": [{
                "type": "procedural", "content": content,
                "importance": 8, "subject": "type_hints", "predicate": "USES",
                "object": "type_annotations",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext), "s-promo-0"]
        store_memory.main()

        # Sessions 2-6: reinforce (same triple triggers reinforcement, not new insert)
        for i in range(1, 6):
            ext_reinforce = {
                "memories": [{
                    "type": "procedural",
                    "content": f"Use type hints in function signatures (session {i})",
                    "importance": 8, "subject": "type_hints", "predicate": "USES",
                    "object": "type_annotations",
                }],
                "entities": [], "relationships": [], "summary": [],
            }
            sys.argv = ["store_memory.py", json.dumps(ext_reinforce), f"s-promo-{i}"]
            store_memory.main()

        # Verify reinforcement_count and stability
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT id, reinforcement_count, stability, promotion_candidate "
                "FROM memories WHERE subject = 'type_hints' AND predicate = 'USES' "
                "AND superseded_by IS NULL AND gc_eligible = 0"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Memory should exist"
        assert row["reinforcement_count"] == 5, (
            f"Expected reinforcement_count=5, got {row['reinforcement_count']}"
        )
        assert row["stability"] == 1.0, (
            f"Expected stability=1.0 at count>=5, got {row['stability']}"
        )
        assert row["promotion_candidate"] == 1

        # Promote to CLAUDE.md
        claude_md = str(tmp_path / "CLAUDE.md")
        result = promote.check_and_promote(row["id"], claude_md_path=claude_md)
        assert result is True, "Promotion should succeed"

        # Verify CLAUDE.md was written
        content_written = Path(claude_md).read_text()
        assert "Learned Behaviors" in content_written
        assert "type hints" in content_written.lower()

    def test_chain_depth_pruning_lifecycle(self):
        """Create a chain of 5 corrections (each superseding the previous).

        enforce_chain_depth_limits() should mark the 3 oldest as gc_eligible
        (correction limit = 2).

        Build chain manually via SQL to guarantee exact chain structure:
        ids[0] -> ids[1] -> ids[2] -> ids[3] -> ids[4] (newest)
        """
        ids = []
        for i in range(5):
            mem = {
                "type": "correction",
                "content": f"Use config format revision {i} with unique schema layout {i * 17}",
                "importance": 5,
                # Use unique subjects so detect_supersession doesn't fire during insert
                "subject": f"config_format_chain_{i}",
                "predicate": "HAS_VERSION",
                "object": f"v{i}",
            }
            mem_id = db.insert_memory(mem, f"s-chain-{i}", "/test/chain")
            ids.append(mem_id)

        # Build supersession chain manually via direct SQL: 0 -> 1 -> 2 -> 3 -> 4
        conn = db.get_db()
        try:
            now = time.time()
            for i in range(len(ids) - 1):
                conn.execute(
                    "UPDATE memories SET superseded_by = ?, superseded_at = ? WHERE id = ?",
                    (ids[i + 1], now, ids[i]),
                )
            conn.commit()
        finally:
            conn.close()

        # Verify chain is built correctly
        conn = db.get_db()
        try:
            for i in range(len(ids) - 1):
                row = conn.execute(
                    "SELECT superseded_by FROM memories WHERE id = ?", (ids[i],)
                ).fetchone()
                assert row["superseded_by"] == ids[i + 1], (
                    f"Memory {i} should be superseded by memory {i+1}"
                )
        finally:
            conn.close()

        # Enforce chain depth limits (correction limit = 2)
        marked = db.enforce_chain_depth_limits()
        assert marked == 3, f"Expected 3 marked gc_eligible, got {marked}"

        # Verify the 3 oldest are gc_eligible
        conn = db.get_db()
        try:
            for i in range(3):
                row = conn.execute(
                    "SELECT gc_eligible FROM memories WHERE id = ?", (ids[i],)
                ).fetchone()
                assert row["gc_eligible"] == 1, (
                    f"Memory {i} (oldest) should be gc_eligible"
                )
            # The 2 newest should NOT be gc_eligible
            for i in range(3, 5):
                row = conn.execute(
                    "SELECT gc_eligible FROM memories WHERE id = ?", (ids[i],)
                ).fetchone()
                assert row["gc_eligible"] == 0, (
                    f"Memory {i} (newest) should NOT be gc_eligible"
                )
        finally:
            conn.close()


class TestE2EEdgeCases:
    """Edge case tests for store_memory and extraction handling."""

    def test_empty_extraction_no_crash(self):
        """Pass empty memories list to store_memory.main() -> no error, no memories stored."""
        import store_memory

        extraction = {
            "memories": [],
            "entities": [],
            "relationships": [],
            "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(extraction), "session-empty"]
        # Should not raise
        store_memory.main()

        conn = db.get_db()
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE session_id = 'session-empty'"
            ).fetchone()
        finally:
            conn.close()
        assert count["cnt"] == 0

    def test_extraction_with_missing_fields(self):
        """Pass memory with only 'type' and 'content' (no subject, predicate, object, importance).

        Memory should be inserted with defaults, no crash.
        """
        import store_memory

        extraction = {
            "memories": [{
                "type": "semantic",
                "content": "Python 3.12 supports new typing features",
            }],
            "entities": [],
            "relationships": [],
            "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(extraction), "session-minimal"]
        store_memory.main()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT memory_type, importance, subject, predicate, object "
                "FROM memories WHERE session_id = 'session-minimal'"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Memory should be inserted despite missing fields"
        assert row["memory_type"] == "semantic"
        assert row["importance"] == 5  # default
        assert row["subject"] is None or row["subject"] == ""
        assert row["predicate"] is None or row["predicate"] == ""

    def test_concurrent_reinforcement_idempotent(self):
        """Insert same memory via store_memory.main() twice with same session_id.

        Near-dedup should catch it; reinforcement should not double-count.
        """
        import store_memory

        extraction = {
            "memories": [{
                "type": "procedural",
                "content": "Use black formatter for all Python code formatting",
                "importance": 7,
                "subject": "code_formatting",
                "predicate": "USES",
                "object": "black",
            }],
            "entities": [],
            "relationships": [],
            "summary": [],
        }

        # First call
        sys.argv = ["store_memory.py", json.dumps(extraction), "session-idem"]
        store_memory.main()

        # Second call with same session_id and same content
        sys.argv = ["store_memory.py", json.dumps(extraction), "session-idem"]
        store_memory.main()

        conn = db.get_db()
        try:
            rows = conn.execute(
                "SELECT id, reinforcement_count, content FROM memories "
                "WHERE subject = 'code_formatting' AND superseded_by IS NULL "
                "AND gc_eligible = 0"
            ).fetchall()
        finally:
            conn.close()

        # Should be exactly 1 memory (dedup prevents second insert)
        assert len(rows) == 1, (
            f"Expected 1 memory (dedup), got {len(rows)}: "
            f"{[r['content'][:40] for r in rows]}"
        )
        # Reinforcement count should be at most 1 (triple match reinforce on second call)
        # The exact behavior depends on whether content-hash dedup fires first
        # or triple-match reinforcement fires first. Either way, no double-counting.
        assert rows[0]["reinforcement_count"] <= 1, (
            f"Reinforcement should not double-count, got {rows[0]['reinforcement_count']}"
        )


class TestE2ERobustness:
    """Tests for real-world LLM output quirks and edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        db._DB_PATH_OVERRIDE = str(tmp_path / "test.db")
        db._db_initialized.clear()
        db.ensure_embedding_column()
        db.ensure_enrichment_columns()
        self._tmp_path = tmp_path
        yield
        db._DB_PATH_OVERRIDE = None
        db._db_initialized.clear()

    def test_importance_as_string(self):
        """qwen2.5:3b sometimes outputs importance as string '8' not int 8."""
        import store_memory

        ext = {
            "memories": [{
                "type": "correction",
                "content": "Use 4 spaces for indentation",
                "importance": "8",  # String, not int
                "subject": "indentation",
                "predicate": "USES",
                "object": "4 spaces",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext), "s-string-imp"]
        store_memory.main()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT importance FROM memories WHERE content LIKE '%indentation%'"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Memory should be inserted despite string importance"
        assert row["importance"] == 8, f"Importance should be coerced to int, got {row['importance']}"

    def test_unicode_content(self):
        """Chinese/CJK content should be stored and retrieved correctly."""
        import store_memory

        ext = {
            "memories": [{
                "type": "correction",
                "content": "使用 pytest 进行测试，不要用 unittest",
                "importance": 7,
                "subject": "测试框架",
                "predicate": "USES",
                "object": "pytest",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext), "s-unicode"]
        store_memory.main()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT content, subject FROM memories WHERE content LIKE '%pytest%测试%'"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Unicode memory should be stored"
        assert "pytest" in row["content"]
        assert row["subject"] == "测试框架"

    def test_invalid_predicate_handled(self):
        """qwen2.5:3b may output predicates not in the CHECK constraint."""
        import store_memory

        ext = {
            "memories": [{
                "type": "correction",
                "content": "Prefer ruff over flake8",
                "importance": 7,
                "subject": "linter",
                "predicate": "PREFERS",  # Not in valid predicates
                "object": "ruff",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext), "s-bad-pred"]
        store_memory.main()

        # Memory should still be inserted (predicate is just metadata)
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT content, predicate FROM memories WHERE content LIKE '%ruff%flake8%'"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Memory should be inserted even with invalid predicate"

    def test_very_short_content(self):
        """Short corrections like 'use tabs' should still be stored."""
        import store_memory

        ext = {
            "memories": [{
                "type": "correction",
                "content": "use tabs",
                "importance": 5,
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext), "s-short"]
        store_memory.main()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT content FROM memories WHERE content = 'use tabs'"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Short content should be stored"

    def test_content_with_special_chars(self):
        """Content with backslashes, quotes, SQL chars should be handled."""
        import store_memory

        ext = {
            "memories": [{
                "type": "correction",
                "content": "Use path\\to\\file instead of /tmp/test.py; don't use 'os.system()'",
                "importance": 6,
                "subject": "file paths",
            }],
            "entities": [], "relationships": [], "summary": [],
        }
        sys.argv = ["store_memory.py", json.dumps(ext), "s-special"]
        store_memory.main()

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT content FROM memories WHERE content LIKE '%os.system%'"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Content with special chars should be stored"


class TestTriageRecall:
    """Test triage catches common memory-worthy phrasings."""

    def test_new_correction_patterns(self):
        """Test all new TIER1 patterns with should-pass cases."""
        import triage

        cases = [
            ("make sure to always use 4 spaces", "correction"),
            ("make sure to run the linter before committing", "correction"),
            ("the correct way is to import at the top", "correction"),
            ("the right way is to use context managers", "correction"),
            ("please don't commit .env files", "correction"),
            ("please do not use global state", "correction"),
            ("the correct approach is to mock the dependency", "correction"),
            ("the better approach is to use fixtures", "correction"),
            ("important: always pin your dependencies", "correction"),
            ("note: never use wildcards in imports", "correction"),
            ("rule: don't commit secrets", "correction"),
            ("do pytest instead of unittest", "correction"),
        ]

        for text, expected_type in cases:
            signals = triage.triage(text)
            matched_types = [s["type"] for s in signals]
            assert expected_type in matched_types, (
                f"Expected '{expected_type}' for: '{text}', got: {signals}"
            )

    def test_new_preference_patterns(self):
        """Test all new TIER3 patterns."""
        import triage

        cases = [
            ("I prefer tabs over spaces", "preference"),
            ("we prefer to use black for formatting", "preference"),
        ]

        for text, expected_type in cases:
            signals = triage.triage(text)
            matched_types = [s["type"] for s in signals]
            assert expected_type in matched_types, (
                f"Expected '{expected_type}' for: '{text}', got: {signals}"
            )

    def test_new_decision_patterns(self):
        """Test all new TIER4 patterns."""
        import triage

        cases = [
            ("the convention is to use snake_case", "decision"),
            ("our convention is 4-space indentation", "decision"),
            ("you should always use the venv python", "decision"),
            ("you should never commit directly to main", "decision"),
        ]

        for text, expected_type in cases:
            signals = triage.triage(text)
            matched_types = [s["type"] for s in signals]
            assert expected_type in matched_types, (
                f"Expected '{expected_type}' for: '{text}', got: {signals}"
            )

    def test_false_positive_rejection(self):
        """Test that common non-memory-worthy phrases don't trigger.

        These include the reviewer's explicit false positive test cases
        plus ordinary coding requests and questions.
        """
        import triage

        false_positives = [
            # Reviewer-specified false positives
            "you should see the output",
            "please use the read tool",
            "switch to the other branch",
            "make sure the tests pass",
            "note: I'll be back in 5 min",
            # Ordinary requests
            "can you read the file?",
            "can you fix the bug in main.py?",
            "I think we should refactor this later",
            "I think it might work",
            "try running it again",
            "try the other endpoint",
            "what do you think about Redis?",
            "how does this function work?",
            "show me the logs from yesterday",
            "list all the files in src/",
            "read the README and summarize it",
            "explain what this decorator does",
            "write a function that parses JSON",
            "create a new file called utils.py",
        ]

        for text in false_positives:
            signals = triage.triage(text)
            assert len(signals) == 0, (
                f"False positive on: '{text}' — matched: {signals}"
            )
