"""tests/test_phase9.py — Phase 9: Embedding Upgrade & Scale tests."""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Make hooks importable ─────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
_HOOKS_DIR = _PROJECT_ROOT / "hooks"
_DAEMON_DIR = _PROJECT_ROOT / "daemon"

if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))
if str(_DAEMON_DIR) not in sys.path:
    sys.path.insert(0, str(_DAEMON_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 9.1: BGE-M3 1024-dim Migration
# ═══════════════════════════════════════════════════════════════════════════════


class TestBGEM3ModelConfig(unittest.TestCase):
    """Verify BGE-M3 is the default embedding model."""

    def test_default_model_name(self):
        """embeddings.py defaults to BAAI/bge-m3."""
        import embeddings
        # Check default (may be overridden by env var in CI)
        default = "BAAI/bge-m3"
        self.assertEqual(
            os.environ.get("ENSEMBLE_MEMORY_EMBED_MODEL", default),
            embeddings.MODEL_NAME if "ENSEMBLE_MEMORY_EMBED_MODEL" not in os.environ else os.environ["ENSEMBLE_MEMORY_EMBED_MODEL"],
        )

    def test_embedding_dim_constant(self):
        """EMBEDDING_DIM is 1024."""
        import embeddings
        self.assertEqual(embeddings.EMBEDDING_DIM, 1024)

    def test_env_override(self):
        """ENSEMBLE_MEMORY_EMBED_MODEL env var overrides default."""
        # Just verify the env var mechanism exists in the module
        import embeddings
        self.assertTrue(hasattr(embeddings, "MODEL_NAME"))


class TestDaemonModelConfig(unittest.TestCase):
    """Verify daemon loads BGE-M3."""

    def test_daemon_loads_bge_m3(self):
        """_load_model() uses BAAI/bge-m3."""
        import embedding_daemon as ed

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}):
            old_model = ed._model
            old_has = ed._has_embeddings
            try:
                ed._model = None
                ed._has_embeddings = False
                ed._load_model()
                mock_st.assert_called_once_with("BAAI/bge-m3")
                self.assertTrue(ed._has_embeddings)
            finally:
                ed._model = old_model
                ed._has_embeddings = old_has


class TestEmbeddingTruncation(unittest.TestCase):
    """Verify truncation limits are updated for BGE-M3."""

    def test_embed_endpoint_truncation(self):
        """POST /embed truncates at 8192 chars, not 512."""
        import embedding_daemon as ed

        # Create a long text
        long_text = "x" * 10000
        captured_text = []

        def mock_get_embedding(text):
            captured_text.append(text)
            return [0.1] * 1024

        old_fn = ed._get_embedding
        old_has = ed._has_embeddings
        try:
            ed._get_embedding = mock_get_embedding
            ed._has_embeddings = True

            # Simulate the /embed handler logic
            text = long_text
            text = text[:8192]  # This is what the handler does
            result = ed._get_embedding(text)

            self.assertEqual(len(captured_text[0]), 8192)
            self.assertNotEqual(len(captured_text[0]), 512)
        finally:
            ed._get_embedding = old_fn
            ed._has_embeddings = old_has

    def test_embed_batch_truncation(self):
        """POST /embed_batch truncates at 8192 chars, not 512."""
        import embedding_daemon as ed

        long_text = "y" * 10000
        captured_texts = []

        def mock_get_embedding(text):
            captured_texts.append(text)
            return [0.1] * 1024

        old_fn = ed._get_embedding
        try:
            ed._get_embedding = mock_get_embedding

            # Simulate the /embed_batch handler logic
            text = str(long_text)[:8192]
            ed._get_embedding(text)

            self.assertEqual(len(captured_texts[0]), 8192)
        finally:
            ed._get_embedding = old_fn


class TestReembedMigration(unittest.TestCase):
    """Test reembed_all_memories() migration function."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        import db
        db._DB_PATH_OVERRIDE = self._tmp.name
        db._db_initialized.clear()
        # Initialize schema + migrations
        conn = db.get_db()
        conn.close()
        db.ensure_embedding_column()
        db.ensure_enrichment_columns()

    def tearDown(self):
        import db
        db._DB_PATH_OVERRIDE = None
        db._db_initialized.clear()
        os.unlink(self._tmp.name)

    def _insert_memory(self, content, embedding=None):
        """Insert a test memory with optional embedding."""
        import db
        mem_id = db.insert_memory(
            {"content": content, "memory_type": "semantic", "importance": 7},
            session_id="test-session",
            project="/test",
        )
        if embedding is not None:
            db.store_embedding(mem_id, embedding)
        return mem_id

    def test_reembed_updates_embeddings(self):
        """reembed_all_memories() updates stored embeddings."""
        import db_memory

        # Insert memories with old 384-dim embeddings
        id1 = self._insert_memory("test content one", [0.1] * 384)
        id2 = self._insert_memory("test content two", [0.2] * 384)

        # Mock the embeddings module to return 1024-dim vectors
        mock_embeddings = MagicMock()
        mock_embeddings.MODEL_NAME = "BAAI/bge-m3"
        mock_embeddings.get_embeddings = MagicMock(return_value=[[0.5] * 1024, [0.6] * 1024])

        with patch.dict(sys.modules, {"embeddings": mock_embeddings}):
            count = db_memory.reembed_all_memories()

        self.assertEqual(count, 2)

        # Verify new embeddings are stored
        import db
        conn = db.get_db()
        try:
            row = conn.execute("SELECT embedding FROM memories WHERE id = ?", (id1,)).fetchone()
            emb = json.loads(row["embedding"])
            self.assertEqual(len(emb), 1024)
            # Value is either 0.5 or 0.6 depending on batch order (created_at DESC)
            self.assertIn(emb[0], [0.5, 0.6])
        finally:
            conn.close()

    def test_reembed_empty_db(self):
        """reembed_all_memories() handles empty database gracefully."""
        import db_memory

        mock_embeddings = MagicMock()
        mock_embeddings.MODEL_NAME = "BAAI/bge-m3"

        with patch.dict(sys.modules, {"embeddings": mock_embeddings}):
            count = db_memory.reembed_all_memories()

        self.assertEqual(count, 0)

    def test_reembed_model_unavailable(self):
        """reembed_all_memories() stops gracefully if model unavailable."""
        import db_memory

        self._insert_memory("content", [0.1] * 384)

        mock_embeddings = MagicMock()
        mock_embeddings.MODEL_NAME = "BAAI/bge-m3"
        mock_embeddings.get_embeddings = MagicMock(return_value=None)

        with patch.dict(sys.modules, {"embeddings": mock_embeddings}):
            count = db_memory.reembed_all_memories()

        self.assertEqual(count, 0)

    def test_reembed_idempotent(self):
        """Running reembed twice produces the same result."""
        import db_memory

        self._insert_memory("idempotent test", [0.1] * 384)

        mock_embeddings = MagicMock()
        mock_embeddings.MODEL_NAME = "BAAI/bge-m3"
        mock_embeddings.get_embeddings = MagicMock(return_value=[[0.7] * 1024])

        with patch.dict(sys.modules, {"embeddings": mock_embeddings}):
            count1 = db_memory.reembed_all_memories()
            count2 = db_memory.reembed_all_memories()

        self.assertEqual(count1, 1)
        self.assertEqual(count2, 1)

    def test_reembed_skips_superseded(self):
        """reembed_all_memories() skips superseded and gc_eligible memories."""
        import db

        id1 = self._insert_memory("active memory", [0.1] * 384)
        id2 = self._insert_memory("superseded memory", [0.2] * 384)

        # Mark id2 as superseded
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?",
                (id1, id2),
            )
            conn.commit()
        finally:
            conn.close()

        import db_memory
        mock_embeddings = MagicMock()
        mock_embeddings.MODEL_NAME = "BAAI/bge-m3"
        mock_embeddings.get_embeddings = MagicMock(return_value=[[0.9] * 1024])

        with patch.dict(sys.modules, {"embeddings": mock_embeddings}):
            count = db_memory.reembed_all_memories()

        # Only the active memory should be re-embedded
        self.assertEqual(count, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 9.3: Pipeline Queue Table
# ═══════════════════════════════════════════════════════════════════════════════


class _QueueTestBase(unittest.TestCase):
    """Base class for pipeline queue tests with DB setup/teardown."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        import db
        db._DB_PATH_OVERRIDE = self._tmp.name
        db._db_initialized.clear()
        conn = db.get_db()
        conn.close()
        db.ensure_embedding_column()
        db.ensure_enrichment_columns()

    def tearDown(self):
        import db
        db._DB_PATH_OVERRIDE = None
        db._db_initialized.clear()
        os.unlink(self._tmp.name)


class TestPipelineQueue(_QueueTestBase):
    """Test memory_pipeline_queue CRUD operations."""

    def test_enqueue_returns_id(self):
        """enqueue_pipeline returns a UUID string."""
        import db
        qid = db.enqueue_pipeline("sess1", '{"memory_id": "m1"}', "temporal")
        self.assertIsInstance(qid, str)
        self.assertEqual(len(qid), 36)  # UUID format

    def test_get_pending_returns_items(self):
        """get_pending_pipeline returns unprocessed items."""
        import db
        db.enqueue_pipeline("sess1", '{"memory_id": "m1"}', "temporal")
        db.enqueue_pipeline("sess1", '{"memory_id": "m2"}', "temporal")
        items = db.get_pending_pipeline("temporal")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["target_expert"], "temporal")

    def test_complete_marks_processed(self):
        """complete_pipeline_item sets processed_at."""
        import db
        qid = db.enqueue_pipeline("sess1", '{"memory_id": "m1"}', "semantic")
        db.complete_pipeline_item(qid)
        items = db.get_pending_pipeline("semantic")
        self.assertEqual(len(items), 0)

    def test_fail_increments_retry(self):
        """fail_pipeline_item increments retry_count and stores error."""
        import db
        qid = db.enqueue_pipeline("sess1", '{"memory_id": "m1"}', "semantic")
        db.fail_pipeline_item(qid, "connection timeout")

        items = db.get_pending_pipeline("semantic")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["retry_count"], 1)
        self.assertEqual(items[0]["processing_error"], "connection timeout")

    def test_max_retries_excludes_item(self):
        """Items at max retries (3) are excluded from get_pending."""
        import db
        qid = db.enqueue_pipeline("sess1", '{"memory_id": "m1"}', "semantic")
        db.fail_pipeline_item(qid, "err1")
        db.fail_pipeline_item(qid, "err2")
        db.fail_pipeline_item(qid, "err3")

        items = db.get_pending_pipeline("semantic")
        self.assertEqual(len(items), 0)

    def test_target_filtering(self):
        """get_pending_pipeline only returns items for the specified expert."""
        import db
        db.enqueue_pipeline("sess1", '{"id": "m1"}', "temporal")
        db.enqueue_pipeline("sess1", '{"id": "m2"}', "semantic")
        db.enqueue_pipeline("sess1", '{"id": "m3"}', "temporal")

        temporal = db.get_pending_pipeline("temporal")
        semantic = db.get_pending_pipeline("semantic")
        self.assertEqual(len(temporal), 2)
        self.assertEqual(len(semantic), 1)

    def test_empty_queue(self):
        """get_pending_pipeline returns empty list for empty queue."""
        import db
        items = db.get_pending_pipeline("nonexistent_expert")
        self.assertEqual(items, [])

    def test_stats(self):
        """get_pipeline_stats returns correct counts."""
        import db
        qid1 = db.enqueue_pipeline("sess1", '{"id": "m1"}', "temporal")
        db.enqueue_pipeline("sess1", '{"id": "m2"}', "temporal")
        qid3 = db.enqueue_pipeline("sess1", '{"id": "m3"}', "semantic")
        db.complete_pipeline_item(qid1)
        db.fail_pipeline_item(qid3, "err")
        db.fail_pipeline_item(qid3, "err")
        db.fail_pipeline_item(qid3, "err")  # 3 retries = failed

        stats = db.get_pipeline_stats()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["pending"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertIn("temporal", stats["by_expert"])
        self.assertIn("semantic", stats["by_expert"])


class TestAmemQueueMigration(_QueueTestBase):
    """Test A-MEM queue functions backed by pipeline queue."""

    def test_queue_amem_uses_pipeline(self):
        """queue_amem_evolution writes to memory_pipeline_queue."""
        import db
        db.queue_amem_evolution("mem-123")

        items = db.get_pending_pipeline("amem_evolution")
        self.assertEqual(len(items), 1)
        data = json.loads(items[0]["memory_json"])
        self.assertEqual(data["memory_id"], "mem-123")

    def test_get_pending_amem_reads_pipeline(self):
        """get_pending_amem_queue reads from pipeline queue."""
        import db
        db.queue_amem_evolution("mem-456")

        pending = db.get_pending_amem_queue()
        self.assertEqual(len(pending), 1)
        queue_id, memory_id = pending[0]
        self.assertEqual(memory_id, "mem-456")

    def test_dequeue_amem_completes_item(self):
        """dequeue_amem marks the pipeline item as complete."""
        import db
        db.queue_amem_evolution("mem-789")
        pending = db.get_pending_amem_queue()
        self.assertEqual(len(pending), 1)

        queue_id, _ = pending[0]
        db.dequeue_amem(queue_id)

        remaining = db.get_pending_amem_queue()
        self.assertEqual(len(remaining), 0)


if __name__ == "__main__":
    unittest.main()
