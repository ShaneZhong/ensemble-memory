#!/usr/bin/env python3
"""Test suite for ensemble memory system — Phase 1.

Run: python3 tests/test_ensemble_memory.py
  or python3 -m pytest tests/test_ensemble_memory.py -v

Tests are grouped by component:
  1. Triage (regex signal detection)
  2. DB (SQLite hub, temporal scoring, supersession, reinforcement)
  3. Store Memory (SQLite + markdown integration)
  4. Session Start (memory loading + context formatting)
  5. End-to-end (full pipeline: triage → extract → store → load)
"""

import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Add hooks dir to path
HOOKS_DIR = Path(__file__).parent.parent / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

import triage
import db
import write_log


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════════════

class TempDirMixin:
    """Creates a temp dir for each test and cleans up after."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ensemble_memory_test_")
        self.db_path = os.path.join(self.tmpdir, "memory.db")
        self.logs_dir = os.path.join(self.tmpdir, "memory")
        self.data_dir = self.tmpdir
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, "logs", "extractions"), exist_ok=True)

        # Override env vars so tests don't touch real data
        self._env_patches = {
            "ENSEMBLE_MEMORY_DIR": self.tmpdir,
            "ENSEMBLE_MEMORY_LOGS": self.logs_dir,
        }
        self._old_env = {}
        for k, v in self._env_patches.items():
            self._old_env[k] = os.environ.get(k)
            os.environ[k] = v

        # Force db.py to reinitialize with test path
        db._DB_PATH_OVERRIDE = self.db_path

    def tearDown(self):
        # Restore env
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        db._DB_PATH_OVERRIDE = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Triage Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriage(unittest.TestCase):
    """Test regex signal detection."""

    def _triage(self, text):
        """Helper: write text to temp file, run triage, return signals."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"Human: {text}\n")
            f.flush()
            path = f.name
        try:
            # Capture stdout
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            sys.argv = ["triage.py", path]
            with redirect_stdout(buf):
                triage.main()
            return json.loads(buf.getvalue())
        finally:
            os.unlink(path)

    def test_tier1_correction_no_dont(self):
        """'no, don't use X' should trigger Tier 1 correction."""
        signals = self._triage("no, don't use MySQL for this project")
        self.assertTrue(any(s["tier"] == 1 for s in signals))

    def test_tier1_correction_actually(self):
        """'actually, ...' should trigger Tier 1 correction."""
        signals = self._triage("actually, we should use PostgreSQL instead")
        self.assertTrue(any(s["tier"] == 1 for s in signals))

    def test_tier1_correction_stop_doing(self):
        """'stop doing X' should trigger Tier 1."""
        signals = self._triage("stop using that deprecated API")
        self.assertTrue(any(s["tier"] == 1 for s in signals))

    def test_tier1_correction_thats_wrong(self):
        """'that's wrong' should trigger Tier 1."""
        signals = self._triage("that's wrong, the port is 8080 not 3000")
        self.assertTrue(any(s["tier"] == 1 for s in signals))

    def test_tier1_correction_instead_use(self):
        """'instead, use X' should trigger Tier 1."""
        signals = self._triage("instead, use the .venv Python")
        self.assertTrue(any(s["tier"] == 1 for s in signals))

    def test_tier4_decision_lets_use(self):
        """'let's use X' should trigger Tier 4 decision."""
        signals = self._triage("let's use SQLite for the database")
        self.assertTrue(any(s["tier"] == 4 for s in signals))

    def test_tier4_decision_we_will(self):
        """'we will use X' should trigger Tier 4."""
        signals = self._triage("we will use Redis for caching")
        self.assertTrue(any(s["tier"] == 4 for s in signals))

    def test_tier4_decision_decided(self):
        """'decided on X' should trigger Tier 4."""
        signals = self._triage("decided on PostgreSQL for the main database")
        self.assertTrue(any(s["tier"] == 4 for s in signals))

    def test_no_signal_ordinary(self):
        """Ordinary request should return empty signals."""
        signals = self._triage("can you read the file at progress.md")
        self.assertEqual(signals, [])

    def test_no_signal_code(self):
        """Code content should not trigger false positives."""
        signals = self._triage("please write a function that returns true")
        self.assertEqual(signals, [])

    def test_no_signal_question(self):
        """Questions about decisions should not trigger."""
        signals = self._triage("what database should we use?")
        self.assertEqual(signals, [])

    def test_case_insensitive(self):
        """Patterns should match regardless of case."""
        signals = self._triage("NO, DON'T USE THAT LIBRARY")
        self.assertTrue(any(s["tier"] == 1 for s in signals))

    def test_only_scans_user_text(self):
        """Should only scan user text, not assistant responses."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Human: hello\n\nAssistant: no, don't use that approach\n")
            f.flush()
            path = f.name
        try:
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            sys.argv = ["triage.py", path]
            with redirect_stdout(buf):
                triage.main()
            signals = json.loads(buf.getvalue())
            self.assertEqual(signals, [])
        finally:
            os.unlink(path)

    def test_multiple_signals(self):
        """A turn with multiple patterns should return multiple signals."""
        signals = self._triage("no, don't use MySQL. Let's use PostgreSQL instead.")
        self.assertTrue(len(signals) >= 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DB Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDB(TempDirMixin, unittest.TestCase):
    """Test SQLite hub operations."""

    def test_db_creation(self):
        """Database and tables should be created on first get_db()."""
        conn = db.get_db()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        self.assertIn("memories", tables)
        self.assertIn("sessions", tables)
        self.assertIn("lambda_base_constants", tables)
        self.assertIn("supersession_depth_limits", tables)
        conn.close()

    def test_lambda_base_seeded(self):
        """Lambda base constants should be seeded on creation."""
        conn = db.get_db()
        rows = conn.execute("SELECT * FROM lambda_base_constants").fetchall()
        types = {r[0] for r in rows}
        self.assertEqual(types, {"procedural", "correction", "semantic", "episodic"})
        conn.close()

    def test_insert_memory_basic(self):
        """insert_memory should create a row and return a UUID."""
        mem = {"type": "correction", "content": "Use PostgreSQL not MySQL", "importance": 8, "confidence": 0.95}
        mem_id = db.insert_memory(mem, "session-1", "/test/project")
        self.assertIsNotNone(mem_id)
        self.assertTrue(len(mem_id) > 10)  # UUID-like

        conn = db.get_db()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
        self.assertIsNotNone(row)
        conn.close()

    def test_insert_memory_type_mapping(self):
        """memory_type should be correctly mapped from 'type' key."""
        mem = {"type": "correction", "content": "test content", "importance": 7}
        mem_id = db.insert_memory(mem, "s1", "/test")
        conn = db.get_db()
        row = conn.execute("SELECT memory_type FROM memories WHERE id = ?", (mem_id,)).fetchone()
        self.assertEqual(row[0], "correction")
        conn.close()

    def test_insert_memory_decay_rate(self):
        """decay_rate should come from lambda_base_constants based on type."""
        for mem_type, expected_lambda in [("procedural", 0.01), ("correction", 0.05), ("semantic", 0.10), ("episodic", 0.16)]:
            mem = {"type": mem_type, "content": f"test {mem_type}", "importance": 5}
            mem_id = db.insert_memory(mem, "s1", "/test")
            conn = db.get_db()
            row = conn.execute("SELECT decay_rate FROM memories WHERE id = ?", (mem_id,)).fetchone()
            self.assertAlmostEqual(row[0], expected_lambda, places=2, msg=f"Wrong decay for {mem_type}")
            conn.close()

    def test_insert_memory_stability(self):
        """stability should be computed as (importance-1)/9.0."""
        mem = {"type": "correction", "content": "test stability", "importance": 10}
        mem_id = db.insert_memory(mem, "s1", "/test")
        conn = db.get_db()
        row = conn.execute("SELECT stability FROM memories WHERE id = ?", (mem_id,)).fetchone()
        self.assertAlmostEqual(row[0], 1.0, places=2)  # (10-1)/9 = 1.0
        conn.close()

    def test_insert_memory_stability_low(self):
        """importance=1 should give stability=0."""
        mem = {"type": "semantic", "content": "low importance", "importance": 1}
        mem_id = db.insert_memory(mem, "s1", "/test")
        conn = db.get_db()
        row = conn.execute("SELECT stability FROM memories WHERE id = ?", (mem_id,)).fetchone()
        self.assertAlmostEqual(row[0], 0.0, places=2)  # (1-1)/9 = 0.0
        conn.close()

    def test_insert_memory_gc_protected(self):
        """importance >= 9 should set gc_protected = 1."""
        mem = {"type": "correction", "content": "critical rule", "importance": 9}
        mem_id = db.insert_memory(mem, "s1", "/test")
        conn = db.get_db()
        row = conn.execute("SELECT gc_protected FROM memories WHERE id = ?", (mem_id,)).fetchone()
        self.assertEqual(row[0], 1)
        conn.close()

    def test_insert_memory_dedup(self):
        """Inserting the same content twice for the same project should return the existing ID."""
        mem = {"type": "semantic", "content": "duplicate test", "importance": 5}
        id1 = db.insert_memory(mem, "s1", "/test")
        id2 = db.insert_memory(mem, "s2", "/test")
        self.assertEqual(id1, id2)

    def test_insert_memory_dedup_different_project(self):
        """Same content in different projects should create separate memories."""
        mem = {"type": "semantic", "content": "same content different project", "importance": 5}
        id1 = db.insert_memory(mem, "s1", "/project-a")
        id2 = db.insert_memory(mem, "s1", "/project-b")
        self.assertNotEqual(id1, id2)

    def test_detect_supersession(self):
        """New memory with same subject+predicate should supersede the old one."""
        old = {"type": "semantic", "content": "uses MySQL", "importance": 6, "subject": "project", "predicate": "USES", "object": "MySQL"}
        new = {"type": "semantic", "content": "uses PostgreSQL", "importance": 7, "subject": "project", "predicate": "USES", "object": "PostgreSQL"}

        old_id = db.insert_memory(old, "s1", "/test")
        new_id = db.insert_memory(new, "s2", "/test")
        superseded = db.detect_supersession(new_id, "project", "USES")

        self.assertEqual(superseded, old_id)

        conn = db.get_db()
        row = conn.execute("SELECT superseded_by FROM memories WHERE id = ?", (old_id,)).fetchone()
        self.assertEqual(row[0], new_id)
        conn.close()

    def test_detect_supersession_no_match(self):
        """No supersession when subject+predicate don't match."""
        mem = {"type": "semantic", "content": "uses PostgreSQL", "importance": 6, "subject": "project", "predicate": "USES"}
        mem_id = db.insert_memory(mem, "s1", "/test")
        result = db.detect_supersession(mem_id, "other_project", "USES")
        self.assertIsNone(result)

    def test_get_memories_for_session_start_importance_filter(self):
        """Should only return memories with importance >= min_importance."""
        db.insert_memory({"type": "correction", "content": "important rule", "importance": 8}, "s1", "/test")
        db.insert_memory({"type": "semantic", "content": "low importance fact", "importance": 4}, "s1", "/test")

        memories = db.get_memories_for_session_start(project="/test", min_importance=7)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "important rule")

    def test_get_memories_excludes_superseded(self):
        """Superseded memories should not be loaded at session start."""
        old_id = db.insert_memory({"type": "correction", "content": "old rule", "importance": 8, "subject": "db", "predicate": "USES"}, "s1", "/test")
        new_id = db.insert_memory({"type": "correction", "content": "new rule", "importance": 8, "subject": "db", "predicate": "USES"}, "s2", "/test")
        db.detect_supersession(new_id, "db", "USES")

        memories = db.get_memories_for_session_start(project="/test", min_importance=7)
        contents = [m["content"] for m in memories]
        self.assertIn("new rule", contents)
        self.assertNotIn("old rule", contents)

    def test_temporal_scoring(self):
        """Newer memories should score higher than older ones (all else equal)."""
        # Insert two memories with different created_at
        conn = db.get_db()
        now = time.time()

        db.insert_memory({"type": "correction", "content": "old memory", "importance": 8}, "s1", "/test")
        # Manually backdate the first memory
        conn.execute("UPDATE memories SET created_at = ? WHERE content = 'old memory'", (now - 86400 * 30,))  # 30 days ago
        conn.commit()

        db.insert_memory({"type": "correction", "content": "new memory", "importance": 8}, "s2", "/test")

        memories = db.get_memories_for_session_start(project="/test", min_importance=7)
        # New memory should be first (higher score)
        self.assertEqual(memories[0]["content"], "new memory")

    def test_record_session(self):
        """record_session should create a session row."""
        now = time.time()
        db.record_session("test-session", now)
        conn = db.get_db()
        row = conn.execute("SELECT * FROM sessions WHERE id = 'test-session'").fetchone()
        self.assertIsNotNone(row)
        conn.close()

    def test_get_reinforcement_count(self):
        """Should count matching procedural memories across sessions."""
        db.insert_memory({"type": "procedural", "content": "always use .venv", "importance": 7, "rule": "use .venv"}, "s1", "/test")
        db.insert_memory({"type": "procedural", "content": "always use .venv python", "importance": 7, "rule": "use .venv"}, "s2", "/test")

        count = db.get_reinforcement_count("use .venv")
        self.assertGreaterEqual(count, 1)

    def test_confidence_fields_separate(self):
        """extraction_confidence and confidence should be separate fields."""
        mem = {"type": "correction", "content": "test confidence", "importance": 7, "confidence": 0.85}
        mem_id = db.insert_memory(mem, "s1", "/test")
        conn = db.get_db()
        row = conn.execute("SELECT extraction_confidence, confidence FROM memories WHERE id = ?", (mem_id,)).fetchone()
        self.assertAlmostEqual(row[0], 0.85)  # extraction_confidence from LLM
        self.assertAlmostEqual(row[1], 1.0)   # retrieval confidence starts at 1.0
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Write Log Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWriteLog(TempDirMixin, unittest.TestCase):
    """Test markdown daily log writer."""

    def test_creates_daily_file(self):
        """Should create YYYY-MM-DD.md file."""
        extraction = {"memories": [{"type": "correction", "content": "test memory", "importance": 7}], "summary": ["test"]}
        sys.argv = ["write_log.py", json.dumps(extraction), "test-session"]
        write_log.main()

        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        md_path = Path(self.logs_dir) / f"{date_str}.md"
        self.assertTrue(md_path.exists())

    def test_markdown_format(self):
        """Output should have date heading, session heading, time heading, and HTML comment."""
        extraction = {"memories": [{"type": "correction", "content": "use PostgreSQL", "importance": 8, "confidence": 0.9}], "summary": ["corrected DB"]}
        sys.argv = ["write_log.py", json.dumps(extraction), "fmt-session"]
        write_log.main()

        from datetime import datetime
        md_path = Path(self.logs_dir) / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        content = md_path.read_text()
        self.assertIn("# 20", content)  # Date heading
        self.assertIn("## Session: fmt-session", content)
        self.assertIn("###", content)  # Time heading
        self.assertIn("<!--", content)  # HTML comment metadata

    def test_dedup_prevents_duplicate(self):
        """Writing the same content twice should not create a duplicate entry."""
        extraction = {"memories": [{"type": "semantic", "content": "PostgreSQL for JSONB", "importance": 6}], "summary": ["noted"]}
        sys.argv = ["write_log.py", json.dumps(extraction), "s1"]
        write_log.main()
        sys.argv = ["write_log.py", json.dumps(extraction), "s2"]
        write_log.main()

        from datetime import datetime
        md_path = Path(self.logs_dir) / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        content = md_path.read_text()
        self.assertEqual(content.count("PostgreSQL for JSONB"), 1)

    def test_different_content_both_written(self):
        """Different memories should both appear."""
        ext1 = {"memories": [{"type": "semantic", "content": "fact one", "importance": 5}], "summary": ["one"]}
        ext2 = {"memories": [{"type": "semantic", "content": "fact two", "importance": 5}], "summary": ["two"]}
        sys.argv = ["write_log.py", json.dumps(ext1), "s1"]
        write_log.main()
        sys.argv = ["write_log.py", json.dumps(ext2), "s1"]
        write_log.main()

        from datetime import datetime
        md_path = Path(self.logs_dir) / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        content = md_path.read_text()
        self.assertIn("fact one", content)
        self.assertIn("fact two", content)

    def test_empty_memories_no_write(self):
        """Empty memories list should not write anything."""
        extraction = {"memories": [], "summary": []}
        sys.argv = ["write_log.py", json.dumps(extraction), "empty-session"]
        write_log.main()

        from datetime import datetime
        md_path = Path(self.logs_dir) / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        # File might not exist or should be empty/minimal
        if md_path.exists():
            self.assertNotIn("## Session: empty-session", md_path.read_text())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Session Start Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionStart(TempDirMixin, unittest.TestCase):
    """Test session start memory loading."""

    def _load_context(self, project="/test", min_importance=7):
        """Helper: load memories and format context like session_start.py does."""
        import session_start
        memories = db.get_memories_for_session_start(project=project, min_importance=min_importance)
        if not memories:
            return {}
        ctx = session_start.format_context(memories, project)
        return {"additionalContext": ctx}

    def test_empty_db_returns_empty_json(self):
        """No memories should return empty JSON."""
        result = self._load_context()
        self.assertEqual(result, {})

    def test_loads_high_importance_memories(self):
        """Should load memories with importance >= threshold."""
        db.insert_memory({"type": "correction", "content": "Never use system Python", "importance": 9}, "s1", "/test")
        db.insert_memory({"type": "procedural", "content": "Always check CLAUDE.md", "importance": 7}, "s1", "/test")

        result = self._load_context()
        self.assertIn("additionalContext", result)
        ctx = result["additionalContext"]
        self.assertIn("Never use system Python", ctx)
        self.assertIn("Always check CLAUDE.md", ctx)

    def test_excludes_low_importance(self):
        """Should NOT load memories below threshold."""
        db.insert_memory({"type": "semantic", "content": "some trivial fact", "importance": 4}, "s1", "/test")

        result = self._load_context()
        self.assertEqual(result, {})

    def test_groups_by_type(self):
        """Context should group corrections and procedural rules separately."""
        db.insert_memory({"type": "correction", "content": "fix A", "importance": 8}, "s1", "/test")
        db.insert_memory({"type": "procedural", "content": "rule B", "importance": 7}, "s1", "/test")

        result = self._load_context()
        ctx = result["additionalContext"]
        self.assertIn("Correction", ctx)

    def test_context_format(self):
        """Context should include importance scores and metadata footer."""
        db.insert_memory({"type": "correction", "content": "important correction", "importance": 9}, "s1", "/test")

        result = self._load_context()
        ctx = result["additionalContext"]
        self.assertIn("MUST FOLLOW", ctx)
        self.assertIn("Memories loaded:", ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(TempDirMixin, unittest.TestCase):
    """End-to-end pipeline tests."""

    def test_store_and_load_roundtrip(self):
        """Memory stored via store_memory should be loadable at session start."""
        import store_memory
        import session_start

        extraction = {"memories": [
            {"type": "correction", "content": "Always use .venv Python", "importance": 8,
             "confidence": 0.95, "subject": "python", "predicate": "USES", "object": ".venv"}
        ], "summary": ["Python path corrected"]}

        sys.argv = ["store_memory.py", json.dumps(extraction), "roundtrip-session"]
        store_memory.main()

        memories = db.get_memories_for_session_start(project=os.getcwd(), min_importance=7)
        self.assertTrue(len(memories) > 0)
        ctx = session_start.format_context(memories, os.getcwd())
        self.assertIn(".venv Python", ctx)

    def test_supersession_excludes_old_from_load(self):
        """After supersession, only the new memory should load."""
        import store_memory
        import session_start

        # Store old fact
        old = {"memories": [{"type": "correction", "content": "use MySQL", "importance": 8, "subject": "db", "predicate": "USES", "object": "MySQL"}], "summary": ["old"]}
        sys.argv = ["store_memory.py", json.dumps(old), "s1"]
        store_memory.main()

        # Store new fact that supersedes
        new = {"memories": [{"type": "correction", "content": "use PostgreSQL", "importance": 8, "subject": "db", "predicate": "USES", "object": "PostgreSQL"}], "summary": ["new"]}
        sys.argv = ["store_memory.py", json.dumps(new), "s2"]
        store_memory.main()

        memories = db.get_memories_for_session_start(project=os.getcwd(), min_importance=7)
        ctx = session_start.format_context(memories, os.getcwd()) if memories else ""
        self.assertIn("PostgreSQL", ctx)
        self.assertNotIn("MySQL", ctx)

    def test_full_pipeline_triage_to_store(self):
        """Triage signal → extraction JSON → store_memory → verify in DB."""
        import store_memory

        # Simulate what stop.sh does: triage detects signal, then extraction result stored
        # (We skip the actual Ollama call and use a mock extraction)
        extraction = {"memories": [
            {"type": "procedural", "content": "Always run tests before committing", "importance": 7,
             "rule": "run tests first", "trigger_condition": "before git commit"}
        ], "summary": ["Testing rule established"]}

        sys.argv = ["store_memory.py", json.dumps(extraction), "pipeline-session"]
        store_memory.main()

        # Verify in SQLite
        conn = db.get_db()
        row = conn.execute("SELECT memory_type, importance, decay_rate FROM memories WHERE content LIKE '%run tests%'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "procedural")
        self.assertEqual(row[1], 7)
        self.assertAlmostEqual(row[2], 0.01)  # procedural lambda_base
        conn.close()

        # Verify in markdown
        from datetime import datetime
        md_path = Path(self.logs_dir) / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        self.assertTrue(md_path.exists())
        self.assertIn("run tests", md_path.read_text())


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Monkey-patch db.py to support test path override
    if not hasattr(db, "_DB_PATH_OVERRIDE"):
        db._DB_PATH_OVERRIDE = None

    original_db_path = db._db_path

    def _test_db_path():
        if db._DB_PATH_OVERRIDE:
            return Path(db._DB_PATH_OVERRIDE)
        return original_db_path()

    db._db_path = _test_db_path

    unittest.main(verbosity=2)
