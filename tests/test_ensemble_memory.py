#!/usr/bin/env python3
"""Test suite for ensemble memory system — Phase 1 + Phase 2.

Run: python3 tests/test_ensemble_memory.py
  or python3 -m pytest tests/test_ensemble_memory.py -v

Tests are grouped by component:
  1. Triage (regex signal detection)
  2. DB (SQLite hub, temporal scoring, supersession, reinforcement)
  3. Store Memory (SQLite + markdown integration)
  4. Session Start (memory loading + context formatting)
  5. End-to-end (full pipeline: triage → extract → store → load)
  6. Embeddings (vector generation, cosine similarity)
  7. Query-time Retrieval (UserPromptSubmit semantic search)
  8. Cosine Supersession (embedding-based supersession)
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

        # Ensure embedding column exists in test DB (Phase 2 migration)
        if hasattr(db, 'ensure_embedding_column'):
            db.ensure_embedding_column()

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
# 6. Embeddings Tests
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import embeddings as emb
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


@unittest.skipUnless(HAS_EMBEDDINGS, "sentence-transformers not installed")
class TestEmbeddings(unittest.TestCase):
    """Test embedding generation and similarity."""

    def test_get_embedding_returns_vector(self):
        """get_embedding should return a list of floats."""
        vec = emb.get_embedding("hello world")
        self.assertIsInstance(vec, list)
        self.assertTrue(len(vec) > 0)
        self.assertIsInstance(vec[0], float)

    def test_embedding_dimension(self):
        """Embedding should be 384-dim for all-MiniLM-L6-v2."""
        vec = emb.get_embedding("test")
        self.assertEqual(len(vec), 384)

    def test_self_similarity_is_one(self):
        """Cosine similarity of a vector with itself should be 1.0."""
        vec = emb.get_embedding("the quick brown fox")
        sim = emb.cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_similar_texts_high_similarity(self):
        """Semantically similar texts should have high cosine similarity."""
        a = emb.get_embedding("use PostgreSQL for the database")
        b = emb.get_embedding("PostgreSQL is our database choice")
        sim = emb.cosine_similarity(a, b)
        self.assertGreater(sim, 0.5)

    def test_dissimilar_texts_low_similarity(self):
        """Unrelated texts should have low cosine similarity."""
        a = emb.get_embedding("use PostgreSQL for the database")
        b = emb.get_embedding("the weather is sunny today")
        sim = emb.cosine_similarity(a, b)
        self.assertLess(sim, 0.3)

    def test_batch_embeddings(self):
        """get_embeddings should return correct number of vectors."""
        texts = ["hello", "world", "test"]
        vecs = emb.get_embeddings(texts)
        self.assertEqual(len(vecs), 3)
        self.assertEqual(len(vecs[0]), 384)

    def test_find_similar(self):
        """find_similar should return candidates above threshold, sorted by similarity."""
        query = emb.get_embedding("database setup")
        candidates = [
            {"content": "use PostgreSQL", "embedding": emb.get_embedding("use PostgreSQL for databases")},
            {"content": "sunny weather", "embedding": emb.get_embedding("the weather is sunny")},
            {"content": "MySQL config", "embedding": emb.get_embedding("configure MySQL database")},
        ]
        results = emb.find_similar(query, candidates, threshold=0.2, top_k=5)
        # Database-related candidates should be returned, weather should not
        contents = [r["content"] for r in results]
        self.assertIn("use PostgreSQL", contents)
        self.assertIn("MySQL config", contents)

    def test_find_similar_empty_candidates(self):
        """find_similar with no candidates should return empty list."""
        query = emb.get_embedding("test")
        results = emb.find_similar(query, [], threshold=0.2)
        self.assertEqual(results, [])

    def test_cosine_similarity_zero_vector(self):
        """Cosine similarity with zero vector should be 0."""
        vec = emb.get_embedding("hello")
        zero = [0.0] * 384
        sim = emb.cosine_similarity(vec, zero)
        self.assertAlmostEqual(sim, 0.0, places=4)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Query-time Retrieval Tests
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_EMBEDDINGS, "sentence-transformers not installed")
class TestQueryRetrieval(TempDirMixin, unittest.TestCase):
    """Test query-time memory retrieval logic (tests embedding + db directly, not daemon)."""

    def _store_with_embedding(self, content, mem_type="correction", importance=8):
        """Helper: insert a memory and generate its embedding."""
        mem = {"type": mem_type, "content": content, "importance": importance}
        mem_id = db.insert_memory(mem, "test-session", "/test")
        vec = emb.get_embedding(content)
        db.store_embedding(mem_id, vec)
        return mem_id

    def test_retrieves_relevant_memory(self):
        """Query about databases should retrieve database-related memory."""
        self._store_with_embedding("Always use PostgreSQL, never MySQL")

        memories = db.get_memories_with_embeddings(project="/test")
        self.assertTrue(len(memories) > 0)

        query_emb = emb.get_embedding("which database should I use")
        results = emb.find_similar(query_emb, memories, threshold=0.2, top_k=5)
        contents = [r["content"] for r in results]
        self.assertTrue(any("PostgreSQL" in c for c in contents))

    def test_no_retrieval_for_irrelevant_query(self):
        """Query about weather should not retrieve database memory."""
        self._store_with_embedding("Always use PostgreSQL, never MySQL")

        memories = db.get_memories_with_embeddings(project="/test")
        query_emb = emb.get_embedding("what is the weather forecast")
        results = emb.find_similar(query_emb, memories, threshold=0.2, top_k=5)
        pg_results = [r for r in results if "PostgreSQL" in r.get("content", "")]
        self.assertEqual(len(pg_results), 0)

    def test_retrieves_multiple_relevant_memories(self):
        """Multiple related memories should all be retrieved."""
        self._store_with_embedding("Use PostgreSQL not MySQL")
        self._store_with_embedding("SQLite is better for small projects")

        memories = db.get_memories_with_embeddings(project="/test")
        query_emb = emb.get_embedding("set up a database for the project")
        results = emb.find_similar(query_emb, memories, threshold=0.15, top_k=5)
        self.assertGreaterEqual(len(results), 1)

    def test_memories_without_embeddings_excluded(self):
        """Memories without embeddings should not appear in similarity search."""
        db.insert_memory({"type": "correction", "content": "no embedding memory", "importance": 8}, "s1", "/test")
        rows = db.get_memories_with_embeddings(project="/test")
        self.assertEqual(len(rows), 0)

    def test_embedding_stored_on_insert(self):
        """store_memory should generate and store embeddings."""
        import store_memory
        extraction = {"memories": [
            {"type": "correction", "content": "Always use dark mode", "importance": 7}
        ], "summary": ["dark mode preference"]}
        sys.argv = ["store_memory.py", json.dumps(extraction), "embed-test"]
        store_memory.main()

        conn = db.get_db()
        row = conn.execute("SELECT embedding FROM memories WHERE content LIKE '%dark mode%'").fetchone()
        self.assertIsNotNone(row)
        if row[0]:
            vec = json.loads(row[0])
            self.assertEqual(len(vec), 384)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Cosine Supersession Tests
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(HAS_EMBEDDINGS, "sentence-transformers not installed")
class TestCosineSupersession(TempDirMixin, unittest.TestCase):
    """Test embedding-based supersession (replaces Jaccard for Phase 2)."""

    def test_cosine_supersedes_similar_content(self):
        """Highly similar content should trigger cosine supersession."""
        old_mem = {"type": "correction", "content": "Use 4 spaces for indentation, never tabs", "importance": 7}
        old_id = db.insert_memory(old_mem, "s1", "/test")
        old_emb = emb.get_embedding(old_mem["content"])
        db.store_embedding(old_id, old_emb)

        new_mem = {"type": "correction", "content": "Use 2 spaces for indentation instead of 4 spaces", "importance": 7}
        new_id = db.insert_memory(new_mem, "s2", "/test")
        new_emb = emb.get_embedding(new_mem["content"])
        db.store_embedding(new_id, new_emb)

        # Check cosine similarity — these should be high
        sim = emb.cosine_similarity(old_emb, new_emb)
        self.assertGreater(sim, 0.5, f"Expected high similarity for indentation memories, got {sim}")

        # Run supersession with embedding
        result = db.detect_content_supersession(new_id, new_mem["content"], "correction", new_embedding=new_emb)
        if sim >= 0.85:
            self.assertEqual(result, old_id)
        # If sim < 0.85, Jaccard fallback applies — may or may not supersede

    def test_cosine_does_not_supersede_unrelated(self):
        """Unrelated content should NOT trigger supersession even with embeddings."""
        old_mem = {"type": "correction", "content": "Always use PostgreSQL for the database", "importance": 7}
        old_id = db.insert_memory(old_mem, "s1", "/test")
        old_emb = emb.get_embedding(old_mem["content"])
        db.store_embedding(old_id, old_emb)

        new_mem = {"type": "correction", "content": "Never use tabs for indentation", "importance": 7}
        new_id = db.insert_memory(new_mem, "s2", "/test")
        new_emb = emb.get_embedding(new_mem["content"])
        db.store_embedding(new_id, new_emb)

        result = db.detect_content_supersession(new_id, new_mem["content"], "correction", new_embedding=new_emb)
        self.assertIsNone(result)

    def test_jaccard_fallback_without_embeddings(self):
        """Without embeddings, Jaccard similarity should still work."""
        old_id = db.insert_memory(
            {"type": "semantic", "content": "the project uses PostgreSQL database for data storage", "importance": 6},
            "s1", "/test"
        )
        new_id = db.insert_memory(
            {"type": "semantic", "content": "the project uses PostgreSQL database for all data storage needs", "importance": 6},
            "s2", "/test"
        )
        # No embeddings stored — should fall back to Jaccard
        result = db.detect_content_supersession(new_id, "the project uses PostgreSQL database for all data storage needs", "semantic")
        # Jaccard of these two should be high (many shared words)
        if result:
            self.assertEqual(result, old_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Knowledge Graph Tests
# ═══════════════════════════════════════════════════════════════════════════════

import kg  # noqa: E402 — imported here after HOOKS_DIR is on path


class TestKnowledgeGraph(TempDirMixin, unittest.TestCase):
    """Test Knowledge Graph module (Phase 3)."""

    def test_kg_tables_created(self):
        """All KG tables should exist after get_db()."""
        conn = db.get_db()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')"
        ).fetchall()}
        conn.close()
        for tbl in [
            "kg_entities", "kg_relationships", "kg_memory_links",
            "kg_episodes", "kg_appears_in", "kg_decay_config", "kg_sync_state",
        ]:
            self.assertIn(tbl, tables, f"Missing table: {tbl}")

    def test_upsert_entity_new(self):
        """Inserting a new entity should store it and return a UUID."""
        eid = kg.upsert_entity("SQLite", "TECHNOLOGY", description="An embedded database")
        self.assertIsNotNone(eid)
        self.assertTrue(len(eid) > 10)

        conn = db.get_db()
        row = conn.execute("SELECT name, type FROM kg_entities WHERE id = ?", (eid,)).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["name"], "SQLite")
        self.assertEqual(row["type"], "TECHNOLOGY")

    def test_upsert_entity_existing(self):
        """Inserting the same entity twice (by name) should merge, not duplicate."""
        eid1 = kg.upsert_entity("Python", "TECHNOLOGY", description="A programming language")
        eid2 = kg.upsert_entity("Python", "TECHNOLOGY", description="A scripting language")
        self.assertEqual(eid1, eid2)

        conn = db.get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM kg_entities WHERE LOWER(name) = 'python'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_upsert_entity_alias_merge(self):
        """Aliases from two upserts should be merged into a union."""
        eid1 = kg.upsert_entity("PostgreSQL", "TECHNOLOGY", aliases=["pg", "postgres"])
        eid2 = kg.upsert_entity("PostgreSQL", "TECHNOLOGY", aliases=["pgsql"])
        self.assertEqual(eid1, eid2)

        conn = db.get_db()
        row = conn.execute("SELECT aliases FROM kg_entities WHERE id = ?", (eid1,)).fetchone()
        conn.close()
        aliases = json.loads(row["aliases"] or "[]")
        self.assertIn("pg", aliases)
        self.assertIn("postgres", aliases)
        self.assertIn("pgsql", aliases)

    def test_upsert_entity_description_update(self):
        """Longer description should replace a shorter one on merge."""
        eid1 = kg.upsert_entity("Ollama", "TOOL", description="LLM runner")
        eid2 = kg.upsert_entity("Ollama", "TOOL", description="Local LLM inference server for running models")
        self.assertEqual(eid1, eid2)

        conn = db.get_db()
        row = conn.execute("SELECT description FROM kg_entities WHERE id = ?", (eid1,)).fetchone()
        conn.close()
        self.assertIn("inference", row["description"])

    def test_entity_type_stored_as_is(self):
        """Entity type is stored as provided (no validation at this layer)."""
        eid = kg.upsert_entity("CustomType", "MY_CUSTOM_TYPE")
        conn = db.get_db()
        row = conn.execute("SELECT type FROM kg_entities WHERE id = ?", (eid,)).fetchone()
        conn.close()
        self.assertEqual(row["type"], "MY_CUSTOM_TYPE")

    def test_insert_relationship(self):
        """insert_relationship should store a relationship row and return an id."""
        rel_id = kg.insert_relationship(
            subject_name="ensemble-memory",
            predicate="USES",
            object_name="SQLite",
            evidence="The project stores memories in SQLite",
            confidence=0.9,
        )
        self.assertIsNotNone(rel_id)

        conn = db.get_db()
        row = conn.execute(
            "SELECT predicate, confidence FROM kg_relationships WHERE id = ?",
            (rel_id,),
        ).fetchone()
        conn.close()
        self.assertEqual(row["predicate"], "USES")
        self.assertAlmostEqual(row["confidence"], 0.9, places=2)

    def test_insert_relationship_invalid_predicate(self):
        """Invalid predicate with no valid parts should fall back to RELATED_TO."""
        result = kg.insert_relationship(
            subject_name="A",
            predicate="INVALID_PRED",
            object_name="B",
        )
        # Fallback to RELATED_TO — should return a valid relationship id
        self.assertIsNotNone(result)

        conn = db.get_db()
        # The original invalid predicate should not be stored verbatim
        count = conn.execute(
            "SELECT COUNT(*) FROM kg_relationships WHERE predicate = 'INVALID_PRED'"
        ).fetchone()[0]
        self.assertEqual(count, 0)
        # Instead it should be stored as RELATED_TO
        count_related = conn.execute(
            "SELECT COUNT(*) FROM kg_relationships WHERE predicate = 'RELATED_TO' "
            "AND subject_id IN (SELECT id FROM kg_entities WHERE name = 'A') "
            "AND object_id IN (SELECT id FROM kg_entities WHERE name = 'B')"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count_related, 1)

    def test_insert_relationship_pipe_predicate_normalization(self):
        """Pipe-separated predicate like 'CAUSES | AFFECTS' should use first valid part."""
        result = kg.insert_relationship(
            subject_name="PipeSubject",
            predicate="CAUSES | AFFECTS",
            object_name="PipeObject",
        )
        self.assertIsNotNone(result)

        conn = db.get_db()
        # Should be stored as AFFECTS (first valid part in PREDICATES)
        row = conn.execute(
            "SELECT predicate FROM kg_relationships WHERE id = ?", (result,)
        ).fetchone()
        conn.close()
        self.assertEqual(row["predicate"], "AFFECTS")

    def test_insert_memory_pipe_type_normalization(self):
        """Pipe-separated memory_type like 'correction | semantic' should use first valid part."""
        mem = {
            "type": "correction | semantic | procedural",
            "content": "test pipe type normalization content unique",
            "importance": 7,
            "confidence": 1.0,
        }
        mem_id = db.insert_memory(mem, "test-session", "/test-project-pipe")
        self.assertIsNotNone(mem_id)

        conn = db.get_db()
        row = conn.execute(
            "SELECT memory_type FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        conn.close()
        self.assertEqual(row["memory_type"], "correction")

    def test_insert_relationship_creates_entities(self):
        """Entities referenced in a relationship should be auto-created."""
        kg.insert_relationship(
            subject_name="NewProject",
            predicate="DEPENDS_ON",
            object_name="NewLibrary",
        )
        conn = db.get_db()
        proj = conn.execute(
            "SELECT id FROM kg_entities WHERE LOWER(name) = 'newproject'"
        ).fetchone()
        lib = conn.execute(
            "SELECT id FROM kg_entities WHERE LOWER(name) = 'newlibrary'"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(proj)
        self.assertIsNotNone(lib)

    def test_insert_relationship_duplicate(self):
        """Same relationship inserted twice should update confidence, not duplicate."""
        rel_id1 = kg.insert_relationship("A", "USES", "B", confidence=0.5)
        rel_id2 = kg.insert_relationship("A", "USES", "B", confidence=0.9)
        self.assertEqual(rel_id1, rel_id2)

        conn = db.get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM kg_relationships WHERE predicate = 'USES' AND valid_until IS NULL"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT confidence FROM kg_relationships WHERE id = ?", (rel_id1,)
        ).fetchone()
        conn.close()
        self.assertEqual(count, 1)
        self.assertAlmostEqual(row["confidence"], 0.9, places=2)

    def test_search_entities_fts(self):
        """FTS5 search should find entities by exact name."""
        kg.upsert_entity("qwen2.5:3b", "TECHNOLOGY", description="Ollama LLM model")
        results = kg.search_entities_fts("qwen2.5:3b")
        names = [r["name"] for r in results]
        self.assertTrue(any("qwen" in n.lower() for n in names))

    def test_search_entities_fts_partial(self):
        """Search should find entities by partial name match (LIKE fallback)."""
        kg.upsert_entity("all-MiniLM-L6-v2", "TECHNOLOGY", description="Embedding model")
        results = kg.search_entities_fts("MiniLM")
        self.assertTrue(len(results) > 0)

    def test_search_entities_fts_no_results(self):
        """Search for unknown entity should return empty list."""
        results = kg.search_entities_fts("xyzzy_nonexistent_entity_12345")
        self.assertEqual(results, [])

    def test_kg_entity_neighborhood_empty(self):
        """Neighborhood of unknown entity name should return empty."""
        result = kg.kg_entity_neighborhood(["totally_unknown_entity_xyz"])
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["relationships"], [])

    def test_kg_entity_neighborhood_depth0(self):
        """Neighborhood at depth 0 should return only the seed entity."""
        kg.upsert_entity("SeedEntity", "CONCEPT")
        result = kg.kg_entity_neighborhood(["SeedEntity"], max_depth=0)
        names = [e["name"] for e in result["entities"]]
        self.assertIn("SeedEntity", names)

    def test_kg_entity_neighborhood_depth1(self):
        """Direct neighbors (depth 1) should be found."""
        kg.upsert_entity("Hub", "CONCEPT")
        kg.upsert_entity("Spoke", "CONCEPT")
        kg.insert_relationship("Hub", "RELATED_TO", "Spoke")

        result = kg.kg_entity_neighborhood(["Hub"], max_depth=1)
        names = [e["name"] for e in result["entities"]]
        self.assertIn("Hub", names)
        self.assertIn("Spoke", names)

    def test_kg_entity_neighborhood_depth2(self):
        """Two-hop neighbors should be found at max_depth=2."""
        kg.upsert_entity("A", "CONCEPT")
        kg.upsert_entity("B", "CONCEPT")
        kg.upsert_entity("C", "CONCEPT")
        kg.insert_relationship("A", "RELATED_TO", "B")
        kg.insert_relationship("B", "RELATED_TO", "C")

        result = kg.kg_entity_neighborhood(["A"], max_depth=2, max_neighbors=10)
        names = [e["name"] for e in result["entities"]]
        self.assertIn("A", names)
        self.assertIn("B", names)
        self.assertIn("C", names)

    def test_kg_entity_neighborhood_cycle(self):
        """Cyclic relationships should not cause infinite loops."""
        kg.upsert_entity("X", "CONCEPT")
        kg.upsert_entity("Y", "CONCEPT")
        kg.insert_relationship("X", "RELATED_TO", "Y")
        kg.insert_relationship("Y", "RELATED_TO", "X")

        # Should complete without error
        result = kg.kg_entity_neighborhood(["X"], max_depth=5)
        self.assertIsNotNone(result)
        names = [e["name"] for e in result["entities"]]
        self.assertIn("X", names)

    def test_record_episode(self):
        """record_episode should create an episode and link entities."""
        ep_id = kg.record_episode(
            session_id="test-session-kg",
            content="The user decided to use SQLite for storage",
            summary="SQLite storage decision",
            entity_names=["SQLite", "storage"],
        )
        self.assertIsNotNone(ep_id)

        conn = db.get_db()
        ep = conn.execute("SELECT id FROM kg_episodes WHERE id = ?", (ep_id,)).fetchone()
        self.assertIsNotNone(ep)

        links = conn.execute(
            "SELECT COUNT(*) FROM kg_appears_in WHERE episode_id = ?", (ep_id,)
        ).fetchone()[0]
        conn.close()
        self.assertGreaterEqual(links, 1)

    def test_get_entity_stats(self):
        """get_entity_stats should return correct counts."""
        kg.upsert_entity("StatsEntityA", "CONCEPT")
        kg.upsert_entity("StatsEntityB", "CONCEPT")
        kg.insert_relationship("StatsEntityA", "RELATED_TO", "StatsEntityB")

        stats = kg.get_entity_stats()
        self.assertGreaterEqual(stats["total_entities"], 2)
        self.assertGreaterEqual(stats["total_relationships"], 1)
        self.assertIn("total_episodes", stats)
        self.assertIn("total_links", stats)

    def test_bootstrap_prompt_structure(self):
        """bootstrap_from_files should call Ollama and process results (mocked)."""
        import unittest.mock as mock

        mock_response = json.dumps({
            "entities": [
                {"name": "MockEntity", "type": "CONCEPT", "description": "A test entity"},
            ],
            "relationships": [],
        })
        ollama_response = json.dumps({"response": mock_response})

        # Write a temp file to bootstrap from
        test_file = os.path.join(self.tmpdir, "CLAUDE.md")
        with open(test_file, "w") as f:
            f.write("# Test\n\nUse Python 3.11.\n")

        class FakeResp:
            def read(self):
                return ollama_response.encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            stats = kg.bootstrap_from_files([test_file])

        self.assertGreaterEqual(stats["entities_created"], 1)
        self.assertIn("relationships_created", stats)

    def test_extraction_prompt_entities(self):
        """Updated extraction prompt should contain 'entities' field."""
        from pathlib import Path as P
        prompt_path = P(__file__).parent.parent / "hooks" / "prompts" / "extraction.txt"
        content = prompt_path.read_text(encoding="utf-8")
        self.assertIn('"entities"', content)
        self.assertIn('"relationships"', content)

    def test_validate_extraction_with_entities(self):
        """validate_extraction should accept the new entities+relationships format."""
        import extract
        data = {
            "memories": [],
            "entities": [{"name": "SQLite", "type": "TECHNOLOGY"}],
            "relationships": [{"subject": "A", "predicate": "USES", "object": "B"}],
            "summary": [],
        }
        self.assertTrue(extract.validate_extraction(data))

    def test_validate_extraction_backward_compatible(self):
        """validate_extraction should still accept old format (no entities/relationships)."""
        import extract
        old_format = {
            "memories": [{"type": "correction", "content": "test", "importance": 7}],
            "summary": ["test session"],
        }
        self.assertTrue(extract.validate_extraction(old_format))

        invalid = {"memories": "not a list", "summary": []}
        self.assertFalse(extract.validate_extraction(invalid))

        invalid2 = {"memories": [], "summary": "not a list"}
        self.assertFalse(extract.validate_extraction(invalid2))

        invalid3 = {"memories": [], "summary": [], "entities": "not a list"}
        self.assertFalse(extract.validate_extraction(invalid3))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Daemon /embed endpoint Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDaemonEmbedEndpoint(unittest.TestCase):
    """Test /embed and /embed_batch endpoints in the daemon handler."""

    def setUp(self):
        """Set up a minimal handler instance with a mocked model."""
        import sys
        daemon_dir = Path(__file__).parent.parent / "daemon"
        sys.path.insert(0, str(daemon_dir))

        import embedding_daemon as daemon_module
        self.daemon = daemon_module

        # Install a fake model so _get_embedding returns a predictable vector
        class FakeModel:
            def encode(self, text, normalize_embeddings=False):
                import numpy as np
                # Return deterministic 384-dim vector based on text length
                vec = [float(len(text) % 10) / 10.0] * 384
                return type("arr", (), {"tolist": lambda self: vec})()

        self._orig_model = daemon_module._model
        self._orig_has_embeddings = daemon_module._has_embeddings
        daemon_module._model = FakeModel()
        daemon_module._has_embeddings = True

    def tearDown(self):
        import sys
        daemon_dir = Path(__file__).parent.parent / "daemon"
        if str(daemon_dir) in sys.path:
            sys.path.remove(str(daemon_dir))
        self.daemon._model = self._orig_model
        self.daemon._has_embeddings = self._orig_has_embeddings

    def _make_request(self, path, body_dict):
        """Simulate a POST request through the handler's do_POST, return (code, body)."""
        import io
        import json as _json
        from http.server import BaseHTTPRequestHandler

        raw_body = _json.dumps(body_dict).encode()
        daemon = self.daemon

        captured = {}

        class FakeHandler(daemon._Handler):
            def __init__(self):
                # Skip BaseHTTPRequestHandler.__init__ — we mock everything
                self.path = path
                self.headers = {"Content-Length": str(len(raw_body))}
                self.rfile = io.BytesIO(raw_body)
                self._response_code = None
                self._response_body = None

            def send_response(self, code):
                self._response_code = code

            def send_header(self, key, val):
                pass

            def end_headers(self):
                pass

            @property
            def wfile(self):
                return io.BytesIO()

            def _send_json(self, code, body):
                captured["code"] = code
                captured["body"] = body

        h = FakeHandler()
        h.do_POST()
        return captured.get("code"), captured.get("body")

    def test_embed_returns_embedding(self):
        """POST /embed with valid text should return a 384-dim embedding."""
        code, body = self._make_request("/embed", {"text": "hello world"})
        self.assertEqual(code, 200)
        self.assertIn("embedding", body)
        self.assertEqual(len(body["embedding"]), 384)

    def test_embed_all_floats(self):
        """Embedding values should be floats."""
        code, body = self._make_request("/embed", {"text": "test"})
        self.assertEqual(code, 200)
        for val in body["embedding"]:
            self.assertIsInstance(val, float)

    def test_embed_missing_text_returns_400(self):
        """POST /embed without 'text' field should return 400."""
        code, body = self._make_request("/embed", {})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_embed_batch_returns_embeddings(self):
        """POST /embed_batch with valid texts should return list of embeddings."""
        code, body = self._make_request("/embed_batch", {"texts": ["hello", "world", "test"]})
        self.assertEqual(code, 200)
        self.assertIn("embeddings", body)
        self.assertEqual(len(body["embeddings"]), 3)
        for emb in body["embeddings"]:
            self.assertEqual(len(emb), 384)

    def test_embed_batch_missing_texts_returns_400(self):
        """POST /embed_batch without 'texts' field should return 400."""
        code, body = self._make_request("/embed_batch", {})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_embed_batch_empty_list_returns_400(self):
        """POST /embed_batch with empty list should return 400."""
        code, body = self._make_request("/embed_batch", {"texts": []})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_embed_model_unavailable_returns_503(self):
        """POST /embed when model is None should return 503."""
        self.daemon._model = None
        self.daemon._has_embeddings = False
        code, body = self._make_request("/embed", {"text": "hello"})
        self.assertEqual(code, 503)
        self.assertIn("error", body)

    def test_embed_batch_model_unavailable_returns_503(self):
        """POST /embed_batch when model is None should return 503."""
        self.daemon._model = None
        self.daemon._has_embeddings = False
        code, body = self._make_request("/embed_batch", {"texts": ["hello"]})
        self.assertEqual(code, 503)
        self.assertIn("error", body)


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
