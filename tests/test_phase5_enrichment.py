#!/usr/bin/env python3
"""Test suite for Phase 5 Contextual Enrichment.

Run: python3 -m pytest tests/test_phase5_enrichment.py -v

Tests cover:
  1. Enrichment validation (_validate_enrichment)
  2. KG path (_enrich_via_kg)
  3. LLM path (_enrich_via_llm)
  4. Top-level enrich_memory logic
  5. Schema migration and storage (db layer)
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add hooks dir to path
HOOKS_DIR = Path(__file__).parent.parent / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

import db
import enrich


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════════════

class TempDirMixin(unittest.TestCase):
    """Creates a temp dir for each test and cleans up after."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ensemble_memory_test_")
        self.db_path = os.path.join(self.tmpdir, "memory.db")
        self.logs_dir = os.path.join(self.tmpdir, "memory")
        self.data_dir = self.tmpdir
        os.makedirs(self.logs_dir, exist_ok=True)

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
        db._db_initialized.clear()

        # Ensure all columns exist
        if hasattr(db, "ensure_embedding_column"):
            db.ensure_embedding_column()
        if hasattr(db, "ensure_enrichment_columns"):
            db.ensure_enrichment_columns()

    def tearDown(self):
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        db._DB_PATH_OVERRIDE = None
        db._db_initialized.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestEnrichmentValidation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichmentValidation(unittest.TestCase):
    """Test _validate_enrichment quality checks."""

    def test_validate_happy_path(self):
        """Well-formed enriched text with novel words passes validation."""
        original = "Use 4 spaces"
        enriched = "Python coding style: Use 4 spaces for indentation following PEP 8"
        self.assertTrue(enrich._validate_enrichment(enriched, original))

    def test_validate_too_short(self):
        """Text shorter than MIN_ENRICHED_WORDS fails validation."""
        original = "Use spaces"
        enriched = "Use spaces"
        self.assertFalse(enrich._validate_enrichment(enriched, original))

    def test_validate_too_long(self):
        """Text exceeding MAX_ENRICHED_WORDS fails validation."""
        # Build a 100-word string with plenty of novel words
        original = "short original"
        # Generate 100 unique words (not in stop words or original)
        words = [f"novelword{i}" for i in range(100)]
        enriched = " ".join(words)
        self.assertFalse(enrich._validate_enrichment(enriched, original))

    def test_validate_first_person(self):
        """Text starting with 'I ' fails validation."""
        original = "Use 4 spaces"
        enriched = "I recommend using 4 spaces for Python indentation per PEP 8 style"
        self.assertFalse(enrich._validate_enrichment(enriched, original))

    def test_validate_no_novelty(self):
        """Text with no novel words beyond original fails validation."""
        original = "use spaces indentation python"
        # Only stop words + original words
        enriched = "use spaces indentation python the a an is in"
        self.assertFalse(enrich._validate_enrichment(enriched, original))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestEnrichViaKG
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichViaKG(unittest.TestCase):
    """Test _enrich_via_kg with mocked kg module."""

    def test_kg_path_with_relationships(self):
        """With >=2 entities and KG returning a prefix, returns enriched text."""
        mock_neighborhood = {
            "entities": [{"id": "1", "name": "Python", "entity_type": "TECHNOLOGY"}],
            "relationships": [],
            "formatted_prefix": "Python (TECHNOLOGY) USES indentation APPLIES_TO PEP8",
        }
        # Use direct patching of kg import inside enrich
        import enrich as enrich_mod
        entity_names = ["Python", "PEP8"]
        kg_mock = MagicMock()
        kg_mock.kg_entity_neighborhood.return_value = mock_neighborhood

        # Mock db to bypass cold-start guard (entity_count >= 50)
        db_mock = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = [100]
        db_mock.get_db.return_value = mock_conn

        with patch.dict("sys.modules", {"kg": kg_mock, "db": db_mock}):
            result = enrich_mod._enrich_via_kg(
                "Python uses 4 spaces for indentation",
                "procedural",
                entity_names,
                "Python",
            )

        self.assertIsNotNone(result)
        self.assertIn("Python", result)
        # Verify entity info context is included
        self.assertIn("procedural", result.lower())

    def test_kg_path_insufficient_entities(self):
        """With only 1 entity name, KG path returns None immediately."""
        result = enrich._enrich_via_kg(
            "Always use 4 spaces",
            "procedural",
            ["Python"],  # only 1 entity
            "Python",
        )
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestEnrichViaLLM
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichViaLLM(unittest.TestCase):
    """Test _enrich_via_llm with mocked urllib."""

    def _make_ollama_response(self, text: str) -> bytes:
        """Build a fake Ollama /api/generate response."""
        return json.dumps({"response": text, "done": True}).encode()

    def test_llm_path_procedural(self):
        """Successful Ollama call returns enriched text."""
        enriched_text = (
            "Python indentation rule: Always use 4 spaces per level. "
            "This follows PEP 8 style guide and ensures consistency across codebases."
        )
        response_bytes = self._make_ollama_response(enriched_text)

        mock_response = MagicMock()
        mock_response.read.return_value = response_bytes
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = enrich._enrich_via_llm(
                "Use 4 spaces for indentation",
                "procedural",
                "Python",
            )

        self.assertIsNotNone(result)
        self.assertIn("Python", result)

    def test_llm_path_ollama_down(self):
        """URLError from Ollama is caught gracefully, returns None."""
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            result = enrich._enrich_via_llm(
                "Use 4 spaces for indentation",
                "procedural",
                "Python",
            )
        self.assertIsNone(result)

    def test_llm_path_unsupported_type(self):
        """Episodic type has no template, returns None without calling Ollama."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = enrich._enrich_via_llm(
                "I worked on the project today",
                "episodic",
                None,
            )
        self.assertIsNone(result)
        mock_urlopen.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestEnrichMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichMemory(unittest.TestCase):
    """Test top-level enrich_memory routing and quality scoring."""

    def test_episodic_skipped(self):
        """Episodic memories are always skipped (already contextual)."""
        result = enrich.enrich_memory(
            content="I worked on fixing the auth bug today",
            memory_type="episodic",
            importance=9,
            entity_names=["auth", "bug"],
            subject="auth",
        )
        self.assertIsNone(result)

    def test_quality_score(self):
        """_compute_quality returns a float in [0, 1]."""
        original = "Use 4 spaces"
        enriched = (
            "Python coding style: Use 4 spaces for indentation. "
            "Following PEP 8 ensures consistent formatting across all Python projects."
        )
        score = enrich._compute_quality(enriched, original, used_kg=False)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_quality_score_kg_bonus(self):
        """KG path gets a higher quality score due to kg_bonus."""
        original = "Use 4 spaces"
        enriched = (
            "Python coding style: Use 4 spaces for indentation. "
            "Following PEP 8 ensures consistent formatting across all Python projects."
        )
        score_no_kg = enrich._compute_quality(enriched, original, used_kg=False)
        score_with_kg = enrich._compute_quality(enriched, original, used_kg=True)
        self.assertGreater(score_with_kg, score_no_kg)

    def test_low_importance_skipped(self):
        """Memories below MIN_ENRICHMENT_IMPORTANCE threshold are skipped."""
        result = enrich.enrich_memory(
            content="Use 4 spaces for indentation",
            memory_type="procedural",
            importance=3,  # below threshold of 6
            entity_names=["Python", "PEP8"],
            subject="Python",
        )
        self.assertIsNone(result)

    def test_compute_quality_edge_cases(self):
        """Quality score should handle edge cases."""
        import enrich
        # Empty enrichment
        q1 = enrich._compute_quality("", "", False)
        self.assertGreaterEqual(q1, 0.0)
        self.assertLessEqual(q1, 1.0)

        # Identical strings (no novelty)
        q2 = enrich._compute_quality("hello world", "hello world", False)
        self.assertLessEqual(q2, 0.8)  # No novelty score

        # KG bonus
        q3 = enrich._compute_quality("enriched context here", "original", True)
        q4 = enrich._compute_quality("enriched context here", "original", False)
        self.assertGreater(q3, q4)  # KG bonus should increase score


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestSchemaAndStorage
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaAndStorage(TempDirMixin):
    """Test Phase 5 DB schema migration and store_enrichment()."""

    def test_ensure_enrichment_columns_idempotent(self):
        """Calling ensure_enrichment_columns() twice raises no error."""
        # First call happens in setUp via ensure_enrichment_columns()
        # Second call should be idempotent
        db.ensure_enrichment_columns()

        # Verify columns exist
        conn = db.get_db()
        try:
            conn.execute("SELECT enriched_text, enrichment_quality FROM memories LIMIT 1")
        finally:
            conn.close()

    def test_store_enrichment_nonexistent_memory(self):
        """store_enrichment on non-existent memory ID should not crash."""
        # This is a no-op UPDATE (0 rows affected) — should not raise
        db.store_enrichment("nonexistent-uuid", "enriched text", 0.8)
        # Verify nothing was inserted
        conn = db.get_db()
        row = conn.execute("SELECT * FROM memories WHERE id = 'nonexistent-uuid'").fetchone()
        conn.close()
        self.assertIsNone(row)

    def test_store_enrichment_updates_row(self):
        """store_enrichment() updates enriched_text and enrichment_quality columns."""
        # Insert a memory row directly
        conn = db.get_db()
        mem_id = "test-enrich-001"
        conn.execute(
            """
            INSERT INTO memories
              (id, content, content_hash, memory_type, importance,
               extraction_confidence, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id,
                "Use 4 spaces for Python indentation",
                "hash_abc123",
                "procedural",
                8,
                0.9,
                1.0,
                1234567890.0,
            ),
        )
        conn.commit()
        conn.close()

        # Store enrichment
        enriched_text = "Python procedural: Use 4 spaces for indentation per PEP 8 style"
        quality = 0.75
        db.store_enrichment(mem_id, enriched_text, quality)

        # Verify the update
        conn = db.get_db()
        row = conn.execute(
            "SELECT enriched_text, enrichment_quality FROM memories WHERE id = ?",
            (mem_id,),
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], enriched_text)
        self.assertAlmostEqual(row[1], quality, places=4)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestEnrichBatch
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichBatch(TempDirMixin):
    """Test enrich_batch() with DB isolation via TempDirMixin."""

    def setUp(self):
        super().setUp()
        db._DB_PATH_OVERRIDE = self.db_path

    def test_enrich_batch_happy_path(self):
        """Batch with 2 qualifying memories enriches at least 1."""
        db.insert_memory(
            {
                "content": "Use Redis for caching layer",
                "type": "procedural",
                "importance": 7,
                "extraction_confidence": 0.9,
                "confidence": 1.0,
            },
            "test-session",
            "test-project",
        )
        db.insert_memory(
            {
                "content": "Always use indexes on foreign keys",
                "type": "semantic",
                "importance": 8,
                "extraction_confidence": 0.9,
                "confidence": 1.0,
            },
            "test-session",
            "test-project",
        )

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"response": "done", "done": true}'

        with patch("enrich.enrich_memory", return_value={"text": "enriched content", "quality": 0.8}) as mock_enrich, \
             patch("urllib.request.urlopen", return_value=mock_response):
            result = enrich.enrich_batch(min_importance=6, limit=10)

        self.assertGreaterEqual(result["processed"], 2)
        self.assertGreaterEqual(result["enriched"], 1)

    def test_enrich_batch_dry_run(self):
        """dry_run=True processes memories but does not write enriched_text to DB."""
        db.insert_memory(
            {
                "content": "Use connection pooling for database access",
                "type": "procedural",
                "importance": 7,
                "extraction_confidence": 0.9,
                "confidence": 1.0,
            },
            "test-session",
            "test-project",
        )

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"response": "done", "done": true}'

        with patch("enrich.enrich_memory", return_value={"text": "test enriched", "quality": 0.5}), \
             patch("urllib.request.urlopen", return_value=mock_response):
            result = enrich.enrich_batch(min_importance=6, limit=10, dry_run=True)

        self.assertGreaterEqual(result["processed"], 1)

        # enriched_text must still be NULL because dry_run skips storage
        conn = db.get_db()
        row = conn.execute(
            "SELECT enriched_text FROM memories WHERE enriched_text IS NOT NULL"
        ).fetchone()
        conn.close()
        self.assertIsNone(row)

    def test_enrich_batch_empty(self):
        """No memories matching min_importance=10 returns zero counts."""
        result = enrich.enrich_batch(min_importance=10, limit=10)
        self.assertEqual(result, {"processed": 0, "enriched": 0, "skipped": 0})


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestStoreMemoryEnrichment
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoreMemoryEnrichment(TempDirMixin):
    """Integration test: _store_to_sqlite triggers enrichment for qualifying memories."""

    def setUp(self):
        super().setUp()
        db._DB_PATH_OVERRIDE = self.db_path

    def test_store_memory_enrichment_integration(self):
        """_store_to_sqlite calls enrich_memory and stores enriched_text in DB."""
        try:
            import store_memory
        except ImportError:
            self.skipTest("store_memory not available")

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"embedding": [0.1, 0.2]}'

        with patch("enrich.enrich_memory", return_value={"text": "enriched version", "quality": 0.75}) as mock_enrich, \
             patch("urllib.request.urlopen", return_value=mock_response):
            store_memory._store_to_sqlite(
                [{"content": "Use Redis for caching", "type": "procedural", "importance": 7}],
                "test-session",
                [{"name": "Redis"}, {"name": "caching"}],
            )

        # enrich_memory must have been called with entity_names containing Redis and caching
        self.assertTrue(mock_enrich.called, "enrich_memory was not called")
        call_kwargs = mock_enrich.call_args
        entity_names_arg = call_kwargs[1].get("entity_names") or call_kwargs[0][3]
        self.assertIn("Redis", entity_names_arg)
        self.assertIn("caching", entity_names_arg)

        # DB must have the enriched_text stored
        conn = db.get_db()
        row = conn.execute(
            "SELECT enriched_text FROM memories WHERE enriched_text IS NOT NULL"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row, "enriched_text was not persisted to DB")
        self.assertEqual(row[0], "enriched version")


if __name__ == "__main__":
    unittest.main()
